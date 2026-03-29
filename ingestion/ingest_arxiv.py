# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 — Ingest arXiv
# MAGIC
# MAGIC **What this notebook does:** Fetches recent AI/ML/CS papers from the arXiv public
# MAGIC API and writes raw data to `stratum_bronze.arxiv_raw` as a Delta table.
# MAGIC
# MAGIC **Inputs:** arXiv Atom API (`/api/query?search_query=cat:cs.AI&sortBy=submittedDate`)
# MAGIC
# MAGIC **Outputs:** `stratum_bronze.arxiv_raw` Delta table
# MAGIC
# MAGIC **Dependencies:** `config.py`, `utils/api_utils.py`, `utils/delta_utils.py`
# MAGIC
# MAGIC **Design note:** arXiv returns XML (Atom format). The `paginate_arxiv` helper parses
# MAGIC entries into flat dicts. Authors and categories are stored as JSON array strings
# MAGIC to keep bronze as close to the raw shape as possible.

# COMMAND ----------

import json
import logging

from pyspark.sql.types import StructType, StructField, StringType

from config import (
    TABLE_NAMES, ARXIV_CATEGORIES, apply_spark_config, setup_logging,
)
from utils.api_utils import paginate_arxiv
from utils.delta_utils import (
    create_all_databases, create_delta_table, write_bronze,
    get_checkpoint, save_checkpoint, generate_batch_id,
)
from utils.quality_utils import compute_quality_metrics, log_quality_to_delta

logger = setup_logging()

# COMMAND ----------

# -- Widget definitions --
dbutils.widgets.text(
    "categories",
    ",".join(ARXIV_CATEGORIES),
    "Comma-separated arXiv categories",
)
dbutils.widgets.text("max_results", "200", "Max papers to fetch per category")
dbutils.widgets.dropdown("run_mode", "incremental", ["incremental", "full"], "Run mode")

CATEGORIES = [c.strip() for c in dbutils.widgets.get("categories").split(",")]
MAX_RESULTS = int(dbutils.widgets.get("max_results"))
RUN_MODE = dbutils.widgets.get("run_mode")

logger.info(
    "arXiv ingestion — categories=%s, max_results=%d, run_mode=%s",
    CATEGORIES, MAX_RESULTS, RUN_MODE,
)

# COMMAND ----------

apply_spark_config(spark)
create_all_databases(spark)

# COMMAND ----------

# -- Bronze table schema --
# WHY: authors and categories are stored as JSON array strings rather than
# exploded rows — this preserves the raw shape at bronze level.
ARXIV_BRONZE_SCHEMA = StructType([
    StructField("arxiv_id", StringType(), False),
    StructField("title", StringType(), True),
    StructField("summary", StringType(), True),
    StructField("authors", StringType(), True),
    StructField("published", StringType(), True),
    StructField("updated", StringType(), True),
    StructField("categories", StringType(), True),
    StructField("link", StringType(), True),
    StructField("_raw_json", StringType(), True),
])

create_delta_table(spark, TABLE_NAMES["arxiv_raw"], ARXIV_BRONZE_SCHEMA)

# COMMAND ----------

def fetch_papers(categories, max_results):
    """Fetch papers from arXiv across multiple categories, deduplicated.

    Args:
        categories: List of arXiv category strings (e.g. ["cs.AI", "cs.LG"]).
        max_results: Maximum papers per category.

    Returns:
        List of unique paper dicts (deduplicated by arxiv_id).
    """
    all_papers = {}

    for category in categories:
        search_query = f"cat:{category}"
        logger.info("Fetching arXiv papers for %s", search_query)

        papers = paginate_arxiv(search_query, max_results=max_results)

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if arxiv_id and arxiv_id not in all_papers:
                all_papers[arxiv_id] = paper

        logger.info(
            "Category %s: %d papers fetched, %d unique total",
            category, len(papers), len(all_papers),
        )

    return list(all_papers.values())

# COMMAND ----------

def build_bronze_df(papers, batch_id):
    """Convert a list of paper dicts into a PySpark DataFrame.

    Args:
        papers: List of dicts from fetch_papers().
        batch_id: The batch identifier.

    Returns:
        PySpark DataFrame matching ARXIV_BRONZE_SCHEMA.
    """
    rows = []
    for p in papers:
        rows.append((
            p.get("arxiv_id", ""),
            p.get("title"),
            p.get("summary"),
            p.get("authors"),       # already JSON string from api_utils
            p.get("published"),
            p.get("updated"),
            p.get("categories"),    # already JSON string from api_utils
            p.get("link"),
            json.dumps(p),
        ))
    return spark.createDataFrame(rows, schema=ARXIV_BRONZE_SCHEMA)

# COMMAND ----------

# -- Main ingestion logic --
batch_id = generate_batch_id("arxiv")
logger.info("Starting arXiv ingestion — batch_id=%s", batch_id)

papers = fetch_papers(CATEGORIES, MAX_RESULTS)

if not papers:
    logger.info("No arXiv papers found — exiting")
    dbutils.notebook.exit("NO_NEW_DATA")

# Incremental filtering: if we have a checkpoint, skip papers published before it
if RUN_MODE == "incremental":
    checkpoint = get_checkpoint(spark, "arxiv")
    last_ts = checkpoint.get("last_timestamp")
    if last_ts:
        original_count = len(papers)
        papers = [p for p in papers if (p.get("published", "") or "") > last_ts]
        logger.info(
            "Incremental filter: %d -> %d papers (last_timestamp=%s)",
            original_count, len(papers), last_ts,
        )

if not papers:
    logger.info("No new papers after incremental filter — exiting")
    dbutils.notebook.exit("NO_NEW_DATA")

df = build_bronze_df(papers, batch_id)

# COMMAND ----------

# -- Data quality check --
quality_metrics = compute_quality_metrics(
    df=df,
    source="arxiv",
    batch_id=batch_id,
    key_columns=["arxiv_id"],
    required_columns=["arxiv_id", "title", "summary", "published"],
)
log_quality_to_delta(spark, quality_metrics)

# COMMAND ----------

# -- Write to Bronze --
rows_written = write_bronze(spark, df, TABLE_NAMES["arxiv_raw"], batch_id, "arxiv")

# -- Update checkpoint with most recent published date --
max_published = max(
    (p.get("published", "") for p in papers),
    default="",
)
save_checkpoint(spark, "arxiv", last_timestamp=max_published, batch_id=batch_id)

logger.info(
    "arXiv ingestion complete — %d rows written, latest published=%s",
    rows_written, max_published,
)

# COMMAND ----------

# -- Verification --
display(spark.table(TABLE_NAMES["arxiv_raw"]).filter(f"_batch_id = '{batch_id}'").limit(5))
print(f"Total rows in {TABLE_NAMES['arxiv_raw']}: {spark.table(TABLE_NAMES['arxiv_raw']).count()}")

# COMMAND ----------

dbutils.notebook.exit(f"SUCCESS: {rows_written} rows ingested")
