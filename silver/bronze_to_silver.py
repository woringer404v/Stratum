# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 — Bronze to Silver
# MAGIC
# MAGIC **What this notebook does:** Reads raw data from all four Bronze tables, applies
# MAGIC per-source transformations (type casting, null handling, deduplication, timestamp
# MAGIC normalisation), and merges everything into a single unified Silver table:
# MAGIC `stratum_silver.tech_signals`.
# MAGIC
# MAGIC **Inputs:** `stratum_bronze.hackernews_raw`, `stratum_bronze.github_raw`,
# MAGIC `stratum_bronze.arxiv_raw`, `stratum_bronze.stackoverflow_raw`
# MAGIC
# MAGIC **Outputs:** `stratum_silver.tech_signals`, `stratum_silver.data_quality_log`
# MAGIC
# MAGIC **Dependencies:** `config.py`, `utils/delta_utils.py`, `utils/quality_utils.py`
# MAGIC
# MAGIC **Unified schema:** `signal_id, source, title, body, url, author, score, created_at,
# MAGIC tags (array<string>), language, _silver_processed_at`

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# MAGIC %run ../utils/delta_utils

# COMMAND ----------

# MAGIC %run ../utils/quality_utils

# COMMAND ----------

import logging

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

logger = setup_logging()

# COMMAND ----------

# -- Widget definitions --
dbutils.widgets.text("sources", "hackernews,github,arxiv,stackoverflow", "Sources to process")
dbutils.widgets.dropdown("reprocess", "false", ["true", "false"], "Reprocess all data")

SOURCES = [s.strip() for s in dbutils.widgets.get("sources").split(",")]
REPROCESS = dbutils.widgets.get("reprocess") == "true"

logger.info("Bronze-to-Silver — sources=%s, reprocess=%s", SOURCES, REPROCESS)

# COMMAND ----------

apply_spark_config(spark)
create_all_databases(spark)

# COMMAND ----------

# ---------------------------------------------------------------------------
# Transform Functions — one per source
# WHY: Each source has a different raw schema. These functions map each to
# the unified Silver schema. Keeping them separate makes debugging and
# testing individual sources straightforward.
# ---------------------------------------------------------------------------

def transform_hackernews(df):
    """Transform bronze.hackernews_raw to the unified Silver schema.

    Mappings:
        - time (unix) -> created_at (timestamp via from_unixtime)
        - by -> author
        - title -> title
        - text -> body
        - url -> url (falls back to HN item URL if null)
        - score -> score
        - signal_id = md5(concat('hackernews', id))
        - Deduplication by id, keeping the row with highest score

    Args:
        df: Bronze DataFrame for hackernews.

    Returns:
        DataFrame conforming to the Silver schema.
    """
    return (
        df
        .withColumn("signal_id", F.md5(F.concat(F.lit("hackernews"), F.col("id").cast("string"))))
        .withColumn("source", F.lit("hackernews"))
        .withColumn("created_at", F.from_unixtime(F.col("time")).cast("timestamp"))
        .withColumn("author", F.col("by"))
        .withColumn(
            "url",
            F.coalesce(
                F.col("url"),
                F.concat(F.lit("https://news.ycombinator.com/item?id="), F.col("id").cast("string")),
            )
        )
        .withColumn("body", F.col("text"))
        .withColumn("tags", F.array().cast(ArrayType(StringType())))
        .withColumn("language", F.lit(None).cast("string"))
        .withColumn("_silver_processed_at", F.current_timestamp())
        # Dedup: keep the row with the highest score per story ID
        .withColumn(
            "_rank",
            F.row_number().over(
                F.Window.partitionBy("id").orderBy(F.col("score").desc(), F.col("_ingested_at").desc())
            )
        )
        .filter(F.col("_rank") == 1)
        .select(
            "signal_id", "source", "title", "body", "url", "author",
            F.col("score").cast("int").alias("score"),
            "created_at", "tags", "language", "_silver_processed_at",
        )
    )


def transform_github(df):
    """Transform bronze.github_raw to the unified Silver schema.

    Mappings:
        - created_at (ISO string) -> created_at (timestamp)
        - name -> title, description -> body, html_url -> url
        - owner_login -> author
        - stargazers_count -> score
        - topics (JSON array string) -> tags (array<string>)
        - language -> language
        - signal_id = md5(concat('github', full_name))
        - Deduplication by full_name, keeping most recently ingested

    Args:
        df: Bronze DataFrame for github.

    Returns:
        DataFrame conforming to the Silver schema.
    """
    return (
        df
        .withColumn("signal_id", F.md5(F.concat(F.lit("github"), F.col("full_name"))))
        .withColumn("source", F.lit("github"))
        .withColumn("created_at", F.to_timestamp(F.col("created_at")))
        .withColumn("author", F.col("owner_login"))
        .withColumn("title", F.col("name"))
        .withColumn("body", F.col("description"))
        .withColumn("url", F.col("html_url"))
        .withColumn("score", F.col("stargazers_count").cast("int"))
        .withColumn(
            "tags",
            F.coalesce(F.from_json(F.col("topics"), ArrayType(StringType())), F.array()),
        )
        .withColumn("_silver_processed_at", F.current_timestamp())
        # Dedup: keep most recently ingested row per repo
        .withColumn(
            "_rank",
            F.row_number().over(
                F.Window.partitionBy("full_name").orderBy(F.col("_ingested_at").desc())
            )
        )
        .filter(F.col("_rank") == 1)
        .select(
            "signal_id", "source", "title", "body", "url", "author",
            "score", "created_at", "tags", "language", "_silver_processed_at",
        )
    )


def transform_arxiv(df):
    """Transform bronze.arxiv_raw to the unified Silver schema.

    Mappings:
        - published (ISO string) -> created_at (timestamp)
        - title -> title, summary -> body, link -> url
        - First author from authors JSON array -> author
        - 0 -> score (arXiv has no engagement score)
        - categories JSON array -> tags
        - signal_id = md5(concat('arxiv', arxiv_id))
        - Deduplication by arxiv_id

    Args:
        df: Bronze DataFrame for arxiv.

    Returns:
        DataFrame conforming to the Silver schema.
    """
    return (
        df
        .withColumn("signal_id", F.md5(F.concat(F.lit("arxiv"), F.col("arxiv_id"))))
        .withColumn("source", F.lit("arxiv"))
        .withColumn("created_at", F.to_timestamp(F.col("published")))
        .withColumn(
            "author",
            F.coalesce(
                F.from_json(F.col("authors"), ArrayType(StringType())).getItem(0),
                F.lit("unknown"),
            )
        )
        .withColumn("title", F.col("title"))
        .withColumn("body", F.col("summary"))
        .withColumn("url", F.col("link"))
        .withColumn("score", F.lit(0).cast("int"))
        .withColumn(
            "tags",
            F.coalesce(F.from_json(F.col("categories"), ArrayType(StringType())), F.array()),
        )
        .withColumn("language", F.lit(None).cast("string"))
        .withColumn("_silver_processed_at", F.current_timestamp())
        # Dedup by arxiv_id, keeping most recently ingested
        .withColumn(
            "_rank",
            F.row_number().over(
                F.Window.partitionBy("arxiv_id").orderBy(F.col("_ingested_at").desc())
            )
        )
        .filter(F.col("_rank") == 1)
        .select(
            "signal_id", "source", "title", "body", "url", "author",
            "score", "created_at", "tags", "language", "_silver_processed_at",
        )
    )


def transform_stackoverflow(df):
    """Transform bronze.stackoverflow_raw to the unified Silver schema.

    Mappings:
        - creation_date (unix) -> created_at (timestamp)
        - title -> title
        - body -> body (HTML stripped via regexp_replace)
        - link -> url
        - owner_display_name -> author
        - score -> score
        - tags JSON array -> tags (array<string>)
        - First tag -> language (heuristic: most SO tags are language-related)
        - signal_id = md5(concat('stackoverflow', question_id))
        - Deduplication by question_id, keeping highest score

    Args:
        df: Bronze DataFrame for stackoverflow.

    Returns:
        DataFrame conforming to the Silver schema.
    """
    return (
        df
        .withColumn("signal_id", F.md5(F.concat(F.lit("stackoverflow"), F.col("question_id").cast("string"))))
        .withColumn("source", F.lit("stackoverflow"))
        .withColumn("created_at", F.from_unixtime(F.col("creation_date")).cast("timestamp"))
        .withColumn("author", F.coalesce(F.col("owner_display_name"), F.lit("anonymous")))
        .withColumn("title", F.col("title"))
        # WHY: SO returns HTML-formatted bodies. Strip tags at Silver to keep
        # bronze raw while making text clean for analytics and LLM input.
        .withColumn(
            "body",
            F.regexp_replace(F.col("body"), "<[^>]+>", " "),
        )
        .withColumn("url", F.col("link"))
        .withColumn(
            "tags",
            F.coalesce(F.from_json(F.col("tags"), ArrayType(StringType())), F.array()),
        )
        .withColumn(
            "language",
            F.coalesce(
                F.from_json(F.col("tags"), ArrayType(StringType())).getItem(0),
                F.lit(None).cast("string"),
            )
        )
        .withColumn("_silver_processed_at", F.current_timestamp())
        # Dedup: keep highest score per question
        .withColumn(
            "_rank",
            F.row_number().over(
                F.Window.partitionBy("question_id").orderBy(F.col("score").desc(), F.col("_ingested_at").desc())
            )
        )
        .filter(F.col("_rank") == 1)
        .select(
            "signal_id", "source", "title", "body", "url", "author",
            F.col("score").cast("int").alias("score"),
            "created_at", "tags", "language", "_silver_processed_at",
        )
    )


# COMMAND ----------

# -- Transform function dispatcher --
TRANSFORM_MAP = {
    "hackernews": {"table": TABLE_NAMES["hackernews_raw"], "fn": transform_hackernews},
    "github": {"table": TABLE_NAMES["github_raw"], "fn": transform_github},
    "arxiv": {"table": TABLE_NAMES["arxiv_raw"], "fn": transform_arxiv},
    "stackoverflow": {"table": TABLE_NAMES["stackoverflow_raw"], "fn": transform_stackoverflow},
}

# COMMAND ----------

# -- Main processing loop --
total_upserted = 0

for source in SOURCES:
    if source not in TRANSFORM_MAP:
        logger.warning("Unknown source '%s' — skipping", source)
        continue

    config = TRANSFORM_MAP[source]
    bronze_table = config["table"]
    transform_fn = config["fn"]
    batch_id = generate_batch_id(f"silver_{source}")

    logger.info("Processing source: %s (bronze table: %s)", source, bronze_table)

    # Read bronze table
    try:
        bronze_df = spark.table(bronze_table)
    except Exception as exc:
        logger.error("Cannot read %s: %s — skipping", bronze_table, exc)
        continue

    # Incremental filtering: only process rows ingested after last silver checkpoint
    if not REPROCESS:
        checkpoint = get_checkpoint(spark, f"silver_{source}")
        last_ts = checkpoint.get("last_timestamp")
        if last_ts:
            bronze_df = bronze_df.filter(F.col("_ingested_at") > last_ts)
            logger.info("Incremental: filtering %s rows after %s", source, last_ts)

    if bronze_df.head(1) is None or bronze_df.count() == 0:
        logger.info("No new bronze data for %s — skipping", source)
        continue

    # Apply source-specific transform
    silver_df = transform_fn(bronze_df)

    # Quality check
    quality_metrics = compute_quality_metrics(
        df=silver_df,
        source=source,
        batch_id=batch_id,
        key_columns=["signal_id"],
        required_columns=["signal_id", "title", "url", "created_at"],
        timestamp_col="created_at",
    )
    log_quality_to_delta(spark, quality_metrics)

    # Upsert to Silver
    result = upsert_to_table(
        spark,
        source_df=silver_df,
        target_table=TABLE_NAMES["tech_signals"],
        merge_keys=["signal_id"],
    )

    upserted = result.get("source_rows", 0)
    total_upserted += upserted

    # Save silver checkpoint with current timestamp
    from datetime import datetime
    save_checkpoint(
        spark, f"silver_{source}",
        last_timestamp=datetime.utcnow().isoformat(),
        batch_id=batch_id,
    )

    logger.info("Source %s: %d rows upserted to silver", source, upserted)

# COMMAND ----------

# -- Summary --
logger.info("Bronze-to-Silver complete — total rows upserted: %d", total_upserted)

tech_signals_count = spark.table(TABLE_NAMES["tech_signals"]).count()
print(f"\nTotal rows in {TABLE_NAMES['tech_signals']}: {tech_signals_count}")

# Per-source breakdown
display(
    spark.table(TABLE_NAMES["tech_signals"])
    .groupBy("source")
    .agg(
        F.count("*").alias("count"),
        F.avg("score").alias("avg_score"),
        F.max("created_at").alias("latest_signal"),
    )
    .orderBy("source")
)

# COMMAND ----------

# -- Sample output --
display(spark.table(TABLE_NAMES["tech_signals"]).orderBy(F.col("created_at").desc()).limit(10))

# COMMAND ----------

dbutils.notebook.exit(f"SUCCESS: {total_upserted} rows upserted to silver")
