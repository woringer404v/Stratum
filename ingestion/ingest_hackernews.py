# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 — Ingest Hacker News
# MAGIC
# MAGIC **What this notebook does:** Fetches top stories from the Hacker News Firebase API and
# MAGIC writes raw data to `stratum_bronze.hackernews_raw` as a Delta table.
# MAGIC
# MAGIC **Inputs:** Hacker News Firebase REST API (`/v0/topstories.json`, `/v0/item/{id}.json`)
# MAGIC
# MAGIC **Outputs:** `stratum_bronze.hackernews_raw` Delta table
# MAGIC
# MAGIC **Dependencies:** `config.py`, `utils/api_utils.py`, `utils/delta_utils.py`
# MAGIC
# MAGIC **Simulated Auto Loader:** Uses checkpoint tracking to only fetch new stories on
# MAGIC incremental runs. HN item IDs are monotonically increasing, so any ID greater than
# MAGIC the checkpoint's `last_id` is new.

# COMMAND ----------

import json
import logging

from pyspark.sql.types import (
    StructType, StructField, LongType, StringType, IntegerType,
)

from config import (
    API_ENDPOINTS, TABLE_NAMES, apply_spark_config, setup_logging,
)
from utils.api_utils import fetch_json
from utils.delta_utils import (
    create_all_databases, create_delta_table, write_bronze,
    get_checkpoint, save_checkpoint, generate_batch_id,
)
from utils.quality_utils import compute_quality_metrics, log_quality_to_delta

logger = setup_logging()

# COMMAND ----------

# -- Widget definitions for notebook parameterisation --
dbutils.widgets.text("batch_size", "200", "Number of top stories to fetch")
dbutils.widgets.dropdown("run_mode", "incremental", ["incremental", "full"], "Run mode")

BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
RUN_MODE = dbutils.widgets.get("run_mode")

logger.info("HackerNews ingestion — batch_size=%d, run_mode=%s", BATCH_SIZE, RUN_MODE)

# COMMAND ----------

# -- Apply Spark config & ensure databases exist --
apply_spark_config(spark)
create_all_databases(spark)

# COMMAND ----------

# -- Bronze table schema --
HN_BRONZE_SCHEMA = StructType([
    StructField("id", LongType(), False),
    StructField("by", StringType(), True),
    StructField("title", StringType(), True),
    StructField("url", StringType(), True),
    StructField("score", IntegerType(), True),
    StructField("time", LongType(), True),
    StructField("type", StringType(), True),
    StructField("descendants", IntegerType(), True),
    StructField("kids", StringType(), True),
    StructField("text", StringType(), True),
    StructField("_raw_json", StringType(), True),
])

create_delta_table(spark, TABLE_NAMES["hackernews_raw"], HN_BRONZE_SCHEMA)

# COMMAND ----------

HN_BASE = API_ENDPOINTS["hackernews"]


def fetch_top_story_ids():
    """Fetch the current list of top story IDs from Hacker News.

    Returns:
        List of integer story IDs (up to ~500 items).
    """
    url = f"{HN_BASE}/topstories.json"
    ids = fetch_json(url)
    logger.info("Fetched %d top story IDs from HN", len(ids))
    return ids


def fetch_single_item(item_id):
    """Fetch a single HN item by ID.

    Args:
        item_id: Integer HN item ID.

    Returns:
        Dict of item data, or None if the item is deleted/missing.
    """
    url = f"{HN_BASE}/item/{item_id}.json"
    try:
        data = fetch_json(url)
        if data is None:
            return None
        return data
    except Exception as exc:
        logger.warning("Failed to fetch HN item %d: %s", item_id, exc)
        return None


def fetch_stories(ids):
    """Fetch item details for a list of HN IDs, filtering to stories only.

    Args:
        ids: List of integer HN item IDs.

    Returns:
        List of story dicts with a _raw_json field appended.
    """
    stories = []
    for item_id in ids:
        item = fetch_single_item(item_id)
        if item is None:
            continue
        # Only keep stories (not jobs, polls, or comments)
        if item.get("type") != "story":
            continue
        item["_raw_json"] = json.dumps(item)
        stories.append(item)

    logger.info("Fetched %d stories from %d IDs", len(stories), len(ids))
    return stories

# COMMAND ----------

def build_bronze_df(stories, batch_id):
    """Convert a list of story dicts into a PySpark DataFrame matching bronze schema.

    Args:
        stories: List of dicts from fetch_stories().
        batch_id: The batch identifier for this ingestion run.

    Returns:
        PySpark DataFrame with columns matching HN_BRONZE_SCHEMA.
    """
    rows = []
    for s in stories:
        rows.append((
            int(s.get("id", 0)),
            s.get("by"),
            s.get("title"),
            s.get("url"),
            s.get("score"),
            s.get("time"),
            s.get("type"),
            s.get("descendants"),
            json.dumps(s.get("kids")) if s.get("kids") else None,
            s.get("text"),
            s.get("_raw_json"),
        ))
    return spark.createDataFrame(rows, schema=HN_BRONZE_SCHEMA)

# COMMAND ----------

# -- Main ingestion logic --
batch_id = generate_batch_id("hackernews")
logger.info("Starting HackerNews ingestion — batch_id=%s", batch_id)

# Fetch top story IDs
all_ids = fetch_top_story_ids()

# Incremental filtering via checkpoint
if RUN_MODE == "incremental":
    checkpoint = get_checkpoint(spark, "hackernews")
    last_id = checkpoint.get("last_id")
    if last_id:
        last_id_int = int(last_id)
        new_ids = [i for i in all_ids if i > last_id_int]
        logger.info(
            "Incremental mode: %d total IDs, %d new (last_id=%s)",
            len(all_ids), len(new_ids), last_id,
        )
        all_ids = new_ids
    else:
        logger.info("No checkpoint found — fetching all %d IDs", len(all_ids))

# Limit to batch_size
ids_to_fetch = all_ids[:BATCH_SIZE]
logger.info("Fetching %d stories", len(ids_to_fetch))

if not ids_to_fetch:
    logger.info("No new stories to ingest — exiting")
    dbutils.notebook.exit("NO_NEW_DATA")

# Fetch story details
stories = fetch_stories(ids_to_fetch)

if not stories:
    logger.info("No valid stories returned — exiting")
    dbutils.notebook.exit("NO_VALID_STORIES")

# Build DataFrame and write to Bronze
df = build_bronze_df(stories, batch_id)

# COMMAND ----------

# -- Data quality check before writing --
quality_metrics = compute_quality_metrics(
    df=df,
    source="hackernews",
    batch_id=batch_id,
    key_columns=["id"],
    required_columns=["id", "title", "score", "time"],
)
log_quality_to_delta(spark, quality_metrics)

# COMMAND ----------

# -- Write to Bronze --
rows_written = write_bronze(spark, df, TABLE_NAMES["hackernews_raw"], batch_id, "hackernews")

# -- Update checkpoint --
max_id = str(max(s["id"] for s in stories))
save_checkpoint(spark, "hackernews", last_id=max_id, batch_id=batch_id)

logger.info(
    "HackerNews ingestion complete — %d rows written, max_id=%s",
    rows_written, max_id,
)

# COMMAND ----------

# -- Verification: display sample and counts --
display(spark.table(TABLE_NAMES["hackernews_raw"]).filter(f"_batch_id = '{batch_id}'").limit(5))
print(f"Total rows in {TABLE_NAMES['hackernews_raw']}: {spark.table(TABLE_NAMES['hackernews_raw']).count()}")

# COMMAND ----------

dbutils.notebook.exit(f"SUCCESS: {rows_written} rows ingested")
