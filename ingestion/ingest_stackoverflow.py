# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 — Ingest Stack Overflow
# MAGIC
# MAGIC **What this notebook does:** Fetches recent questions from the Stack Exchange API v2.3
# MAGIC for technology-related tags and writes raw data to `stratum_bronze.stackoverflow_raw`.
# MAGIC
# MAGIC **Inputs:** Stack Exchange API (`/questions?tagged=...&site=stackoverflow`)
# MAGIC
# MAGIC **Outputs:** `stratum_bronze.stackoverflow_raw` Delta table
# MAGIC
# MAGIC **Dependencies:** `config.py`, `utils/api_utils.py`, `utils/delta_utils.py`
# MAGIC
# MAGIC **Design note:** SO API has a 300 req/day unauthenticated quota. The pagination helper
# MAGIC monitors `quota_remaining` and stops early to preserve headroom for reruns. Body is
# MAGIC returned as HTML — we store it raw here; HTML stripping happens in the Silver layer.

# COMMAND ----------

import json
import logging

from pyspark.sql.types import (
    StructType, StructField, LongType, StringType, IntegerType, BooleanType,
)

from config import (
    TABLE_NAMES, STACKOVERFLOW_TAGS, apply_spark_config, setup_logging,
)
from utils.api_utils import paginate_stackoverflow
from utils.delta_utils import (
    create_all_databases, create_delta_table, write_bronze,
    get_checkpoint, save_checkpoint, generate_batch_id,
)
from utils.quality_utils import compute_quality_metrics, log_quality_to_delta

logger = setup_logging()

# COMMAND ----------

# -- Widget definitions --
dbutils.widgets.text(
    "tags",
    ";".join(STACKOVERFLOW_TAGS),
    "Semicolon-separated SO tags to fetch",
)
dbutils.widgets.text("max_pages", "2", "Max pages per tag")
dbutils.widgets.dropdown("run_mode", "incremental", ["incremental", "full"], "Run mode")

# WHY: We fetch each tag separately (not combined with ;) to get broader coverage.
# Combined tags return questions matching ALL tags; separate fetches get questions
# matching ANY tag, which yields more diverse signals.
TAGS = [t.strip() for t in dbutils.widgets.get("tags").split(";")]
MAX_PAGES = int(dbutils.widgets.get("max_pages"))
RUN_MODE = dbutils.widgets.get("run_mode")

logger.info(
    "StackOverflow ingestion — tags=%s, max_pages=%d, run_mode=%s",
    TAGS, MAX_PAGES, RUN_MODE,
)

# COMMAND ----------

apply_spark_config(spark)
create_all_databases(spark)

# COMMAND ----------

# -- Bronze table schema --
SO_BRONZE_SCHEMA = StructType([
    StructField("question_id", LongType(), False),
    StructField("title", StringType(), True),
    StructField("body", StringType(), True),
    StructField("tags", StringType(), True),
    StructField("score", IntegerType(), True),
    StructField("view_count", IntegerType(), True),
    StructField("answer_count", IntegerType(), True),
    StructField("is_answered", BooleanType(), True),
    StructField("creation_date", LongType(), True),
    StructField("link", StringType(), True),
    StructField("owner_display_name", StringType(), True),
    StructField("owner_reputation", IntegerType(), True),
    StructField("_raw_json", StringType(), True),
])

create_delta_table(spark, TABLE_NAMES["stackoverflow_raw"], SO_BRONZE_SCHEMA)

# COMMAND ----------

def fetch_questions(tags, max_pages):
    """Fetch questions from Stack Overflow across multiple tags, deduplicated.

    Args:
        tags: List of individual tag strings.
        max_pages: Maximum pages to fetch per tag.

    Returns:
        List of unique question dicts (deduplicated by question_id).
    """
    all_questions = {}

    for tag in tags:
        logger.info("Fetching SO questions for tag: %s", tag)

        questions = paginate_stackoverflow(tag, max_pages=max_pages, pagesize=100)

        for q in questions:
            qid = q.get("question_id")
            if qid and qid not in all_questions:
                all_questions[qid] = q

        logger.info(
            "Tag '%s': %d questions fetched, %d unique total",
            tag, len(questions), len(all_questions),
        )

    return list(all_questions.values())

# COMMAND ----------

def build_bronze_df(questions, batch_id):
    """Convert a list of question dicts into a PySpark DataFrame.

    Args:
        questions: List of dicts from fetch_questions().
        batch_id: The batch identifier.

    Returns:
        PySpark DataFrame matching SO_BRONZE_SCHEMA.
    """
    rows = []
    for q in questions:
        owner = q.get("owner", {}) or {}
        rows.append((
            int(q.get("question_id", 0)),
            q.get("title"),
            q.get("body"),
            json.dumps(q.get("tags", [])),
            q.get("score"),
            q.get("view_count"),
            q.get("answer_count"),
            q.get("is_answered"),
            q.get("creation_date"),
            q.get("link"),
            owner.get("display_name"),
            owner.get("reputation"),
            json.dumps(q),
        ))
    return spark.createDataFrame(rows, schema=SO_BRONZE_SCHEMA)

# COMMAND ----------

# -- Main ingestion logic --
batch_id = generate_batch_id("stackoverflow")
logger.info("Starting StackOverflow ingestion — batch_id=%s", batch_id)

questions = fetch_questions(TAGS, MAX_PAGES)

if not questions:
    logger.info("No SO questions found — exiting")
    dbutils.notebook.exit("NO_NEW_DATA")

# Incremental filtering by creation_date
if RUN_MODE == "incremental":
    checkpoint = get_checkpoint(spark, "stackoverflow")
    last_ts = checkpoint.get("last_timestamp")
    if last_ts:
        try:
            last_ts_int = int(last_ts)
            original_count = len(questions)
            questions = [q for q in questions if (q.get("creation_date", 0) or 0) > last_ts_int]
            logger.info(
                "Incremental filter: %d -> %d questions (last_timestamp=%s)",
                original_count, len(questions), last_ts,
            )
        except ValueError:
            logger.warning("Invalid checkpoint timestamp '%s' — fetching all", last_ts)

if not questions:
    logger.info("No new questions after incremental filter — exiting")
    dbutils.notebook.exit("NO_NEW_DATA")

df = build_bronze_df(questions, batch_id)

# COMMAND ----------

# -- Data quality check --
quality_metrics = compute_quality_metrics(
    df=df,
    source="stackoverflow",
    batch_id=batch_id,
    key_columns=["question_id"],
    required_columns=["question_id", "title", "score", "creation_date"],
)
log_quality_to_delta(spark, quality_metrics)

# COMMAND ----------

# -- Write to Bronze --
rows_written = write_bronze(spark, df, TABLE_NAMES["stackoverflow_raw"], batch_id, "stackoverflow")

# -- Update checkpoint with most recent creation_date --
max_creation = str(max(
    (q.get("creation_date", 0) for q in questions),
    default=0,
))
save_checkpoint(spark, "stackoverflow", last_timestamp=max_creation, batch_id=batch_id)

logger.info(
    "StackOverflow ingestion complete — %d rows written, latest creation_date=%s",
    rows_written, max_creation,
)

# COMMAND ----------

# -- Verification --
display(spark.table(TABLE_NAMES["stackoverflow_raw"]).filter(f"_batch_id = '{batch_id}'").limit(5))
print(f"Total rows in {TABLE_NAMES['stackoverflow_raw']}: {spark.table(TABLE_NAMES['stackoverflow_raw']).count()}")

# COMMAND ----------

dbutils.notebook.exit(f"SUCCESS: {rows_written} rows ingested")
