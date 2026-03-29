# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 06 — Silver to Gold
# MAGIC
# MAGIC **What this notebook does:** Computes four analytics-ready Gold tables from the
# MAGIC unified Silver layer using PySpark window functions and aggregations.
# MAGIC
# MAGIC **Inputs:** `stratum_silver.tech_signals`
# MAGIC
# MAGIC **Outputs:**
# MAGIC - `stratum_gold.tech_term_frequency` — term counts per source per day
# MAGIC - `stratum_gold.trending_signals` — top signals by composite score
# MAGIC - `stratum_gold.source_summary` — daily summary per source
# MAGIC - `stratum_gold.tech_velocity` — week-over-week velocity per tech term
# MAGIC
# MAGIC **Dependencies:** `config.py`, `utils/delta_utils.py`

# COMMAND ----------

import logging
from datetime import datetime, timedelta

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import ArrayType, StringType

from config import TABLE_NAMES, apply_spark_config, setup_logging
from utils.delta_utils import (
    create_all_databases, upsert_to_table, optimize_table,
)

logger = setup_logging()

# COMMAND ----------

# -- Widget definitions --
dbutils.widgets.text(
    "date_from",
    (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
    "Start date (YYYY-MM-DD)",
)
dbutils.widgets.text(
    "date_to",
    datetime.utcnow().strftime("%Y-%m-%d"),
    "End date (YYYY-MM-DD)",
)

DATE_FROM = dbutils.widgets.get("date_from")
DATE_TO = dbutils.widgets.get("date_to")

logger.info("Silver-to-Gold — date_from=%s, date_to=%s", DATE_FROM, DATE_TO)

# COMMAND ----------

apply_spark_config(spark)
create_all_databases(spark)

# COMMAND ----------

# -- Read Silver signals for the date range --
silver_df = (
    spark.table(TABLE_NAMES["tech_signals"])
    .withColumn("date", F.to_date("created_at"))
    .filter(
        (F.col("date") >= DATE_FROM) & (F.col("date") <= DATE_TO)
    )
)

signal_count = silver_df.count()
logger.info("Loaded %d silver signals for date range %s to %s", signal_count, DATE_FROM, DATE_TO)

if signal_count == 0:
    logger.warning("No signals in date range — exiting")
    dbutils.notebook.exit("NO_DATA_IN_RANGE")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Gold Table 1: tech_term_frequency
# WHY: Term frequency by source and day enables trend detection, cross-source
# comparison, and powers the velocity calculation.
# ---------------------------------------------------------------------------
def build_term_frequency(silver_df):
    """Compute term frequency per source per day.

    Extracts terms from tags array and title keywords. Groups by term, source,
    and date. Ranks within each source+date partition by frequency.

    Args:
        silver_df: Silver signals DataFrame with a 'date' column.

    Returns:
        DataFrame with columns: term, source, date, frequency, rank_in_source.
    """
    # Extract terms from tags (explode array)
    tag_terms = (
        silver_df
        .select("source", "date", F.explode("tags").alias("term"))
        .filter(F.col("term").isNotNull() & (F.length("term") > 1))
        .withColumn("term", F.lower(F.trim("term")))
    )

    # Extract significant words from titles (>= 4 chars, not stopwords)
    # WHY: Title keywords complement tags — many signals lack tags but have
    # descriptive titles. The 4-char minimum filters out noise.
    stopwords = ["that", "this", "with", "from", "what", "have", "your", "will",
                 "about", "they", "been", "more", "when", "than", "just", "like",
                 "does", "into", "some", "only", "other", "also", "there", "which"]
    stopwords_lit = F.array(*[F.lit(w) for w in stopwords])

    title_terms = (
        silver_df
        .select("source", "date", F.explode(F.split(F.lower("title"), r"\s+")).alias("term"))
        .filter(F.col("term").isNotNull() & (F.length("term") >= 4))
        .withColumn("term", F.regexp_replace("term", r"[^a-z0-9\+\#]", ""))
        .filter(F.length("term") >= 3)
        .filter(~F.array_contains(stopwords_lit, F.col("term")))
    )

    # Union tag terms and title terms, count, and rank
    all_terms = tag_terms.union(title_terms)

    term_freq = (
        all_terms
        .groupBy("term", "source", "date")
        .agg(F.count("*").alias("frequency"))
    )

    # Rank within each source+date by frequency
    window_spec = Window.partitionBy("source", "date").orderBy(F.col("frequency").desc())
    term_freq = term_freq.withColumn("rank_in_source", F.row_number().over(window_spec))

    return term_freq


logger.info("Building tech_term_frequency...")
term_freq_df = build_term_frequency(silver_df)
upsert_to_table(
    spark, term_freq_df,
    TABLE_NAMES["tech_term_frequency"],
    merge_keys=["term", "source", "date"],
)
logger.info("tech_term_frequency: %d rows", term_freq_df.count())

# COMMAND ----------

# ---------------------------------------------------------------------------
# Gold Table 2: trending_signals
# WHY: A composite score combining raw engagement, tag richness, and recency
# gives a more nuanced ranking than any single metric. This is the table
# an SE would demo to show "what's hot in tech right now".
# ---------------------------------------------------------------------------
def build_trending_signals(silver_df):
    """Compute top trending signals by composite score.

    composite_score = score * log(2 + tag_count) * recency_weight
    recency_weight = 1 / (1 + days_since_created)

    Ranks per source, takes top 50 per source per day.

    Args:
        silver_df: Silver signals DataFrame with a 'date' column.

    Returns:
        DataFrame with trending signal columns.
    """
    today = F.current_date()

    trending = (
        silver_df
        .withColumn("tag_count", F.size("tags"))
        .withColumn("days_since", F.datediff(today, F.col("created_at")).cast("double"))
        .withColumn(
            "recency_weight",
            F.lit(1.0) / (F.lit(1.0) + F.col("days_since")),
        )
        .withColumn(
            "composite_score",
            F.col("score") * F.log(F.lit(2.0) + F.col("tag_count")) * F.col("recency_weight"),
        )
    )

    # Rank per source per date, keep top 50
    window_spec = Window.partitionBy("source", "date").orderBy(F.col("composite_score").desc())
    trending = (
        trending
        .withColumn("rank_in_source", F.row_number().over(window_spec))
        .filter(F.col("rank_in_source") <= 50)
        .select(
            "signal_id", "source", "title", "url",
            F.col("score").alias("score"),
            "composite_score", "rank_in_source", "date", "tags",
        )
    )

    return trending


logger.info("Building trending_signals...")
trending_df = build_trending_signals(silver_df)
upsert_to_table(
    spark, trending_df,
    TABLE_NAMES["trending_signals"],
    merge_keys=["signal_id", "date"],
)
logger.info("trending_signals: %d rows", trending_df.count())

# COMMAND ----------

# ---------------------------------------------------------------------------
# Gold Table 3: source_summary
# WHY: Per-source daily summary is the operational health dashboard for the
# pipeline. It answers: "Is each source delivering data? At what volume and
# quality?" Also useful for cross-source comparison analytics.
# ---------------------------------------------------------------------------
def build_source_summary(silver_df):
    """Compute daily summary statistics per data source.

    Args:
        silver_df: Silver signals DataFrame with a 'date' column.

    Returns:
        DataFrame with source, date, signal_count, avg_score, max_score,
        top_tag, unique_authors, avg_body_length.
    """
    # Explode tags to find the mode (most common) tag per source per day
    tag_mode = (
        silver_df
        .select("source", "date", F.explode("tags").alias("tag"))
        .groupBy("source", "date", "tag")
        .agg(F.count("*").alias("tag_count"))
    )
    tag_window = Window.partitionBy("source", "date").orderBy(F.col("tag_count").desc())
    top_tags = (
        tag_mode
        .withColumn("_rank", F.row_number().over(tag_window))
        .filter(F.col("_rank") == 1)
        .select("source", "date", F.col("tag").alias("top_tag"))
    )

    summary = (
        silver_df
        .groupBy("source", "date")
        .agg(
            F.count("*").alias("signal_count"),
            F.round(F.avg("score"), 2).alias("avg_score"),
            F.max("score").alias("max_score"),
            F.countDistinct("author").alias("unique_authors"),
            F.round(F.avg(F.length("body")), 1).alias("avg_body_length"),
        )
    )

    # Join top tag
    summary = summary.join(top_tags, on=["source", "date"], how="left")

    return summary


logger.info("Building source_summary...")
summary_df = build_source_summary(silver_df)
upsert_to_table(
    spark, summary_df,
    TABLE_NAMES["source_summary"],
    merge_keys=["source", "date"],
)
logger.info("source_summary: %d rows", summary_df.count())

# COMMAND ----------

# ---------------------------------------------------------------------------
# Gold Table 4: tech_velocity
# WHY: Velocity (week-over-week change) is the most useful signal for early
# trend detection. A term with 5 mentions this week and 0 last week is more
# interesting than one with 1000 both weeks. The is_accelerating flag makes
# this queryable without computing velocity at read time.
# ---------------------------------------------------------------------------
def build_tech_velocity(silver_df):
    """Compute week-over-week velocity for each tech term.

    velocity = (this_week_count - last_week_count) / max(last_week_count, 1)
    is_accelerating = velocity > 0.2

    Args:
        silver_df: Silver signals DataFrame with a 'date' column.

    Returns:
        DataFrame with term, week_start, this_week_count, last_week_count,
        velocity, is_accelerating.
    """
    # Extract terms (same logic as term_frequency)
    terms = (
        silver_df
        .select("source", "date", F.explode("tags").alias("term"))
        .filter(F.col("term").isNotNull() & (F.length("term") > 1))
        .withColumn("term", F.lower(F.trim("term")))
        .withColumn("week_start", F.date_trunc("week", "date").cast("date"))
    )

    # Count per term per week
    weekly_counts = (
        terms
        .groupBy("term", "week_start")
        .agg(F.count("*").alias("this_week_count"))
    )

    # Self-join to get last week's count
    last_week = (
        weekly_counts
        .withColumnRenamed("this_week_count", "last_week_count")
        .withColumn("week_start", F.date_add("week_start", 7))
    )

    velocity = (
        weekly_counts
        .join(last_week, on=["term", "week_start"], how="left")
        .fillna(0, subset=["last_week_count"])
        .withColumn(
            "velocity",
            (F.col("this_week_count") - F.col("last_week_count"))
            / F.greatest(F.col("last_week_count"), F.lit(1)),
        )
        .withColumn("is_accelerating", F.col("velocity") > 0.2)
    )

    return velocity


logger.info("Building tech_velocity...")
velocity_df = build_tech_velocity(silver_df)
upsert_to_table(
    spark, velocity_df,
    TABLE_NAMES["tech_velocity"],
    merge_keys=["term", "week_start"],
)
logger.info("tech_velocity: %d rows", velocity_df.count())

# COMMAND ----------

# -- OPTIMIZE + ZORDER all Gold tables --
logger.info("Running OPTIMIZE + ZORDER on Gold tables...")

optimize_table(spark, TABLE_NAMES["tech_term_frequency"], zorder_cols=["term", "source"])
optimize_table(spark, TABLE_NAMES["trending_signals"], zorder_cols=["source", "composite_score"])
optimize_table(spark, TABLE_NAMES["source_summary"], zorder_cols=["source"])
optimize_table(spark, TABLE_NAMES["tech_velocity"], zorder_cols=["term"])

logger.info("Gold table optimization complete")

# COMMAND ----------

# -- Summary --
for name in ["tech_term_frequency", "trending_signals", "source_summary", "tech_velocity"]:
    count = spark.table(TABLE_NAMES[name]).count()
    print(f"{TABLE_NAMES[name]}: {count} rows")

# COMMAND ----------

dbutils.notebook.exit("SUCCESS: All Gold tables built")
