# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stratum — Data Quality Utilities
# MAGIC
# MAGIC Functions for computing data quality metrics (nulls, duplicates, freshness),
# MAGIC logging them to a Delta table, and enforcing soft quality gates.
# MAGIC
# MAGIC **Usage:** Other notebooks include this via `%run ../utils/quality_utils`
# MAGIC
# MAGIC **Depends on:** `%run ../config` (must be run first by the calling notebook)

# COMMAND ----------

import json
import logging
import uuid
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, BooleanType, TimestampType,
)

# WHY: No import from config — TABLE_NAMES is already in scope because
# the calling notebook runs %run ../config first.

logger = logging.getLogger("stratum.quality")

# COMMAND ----------

# -- Schema for the Data Quality Log Table --
DATA_QUALITY_LOG_SCHEMA = StructType([
    StructField("log_id", StringType(), False),
    StructField("source", StringType(), False),
    StructField("batch_id", StringType(), False),
    StructField("total_rows", IntegerType(), False),
    StructField("null_counts", StringType(), True),      # JSON dict
    StructField("duplicate_count", IntegerType(), True),
    StructField("freshness_hours", DoubleType(), True),
    StructField("is_fresh", BooleanType(), True),
    StructField("quality_passed", BooleanType(), True),
    StructField("checked_at", TimestampType(), False),
])

# COMMAND ----------

# -- Individual Quality Checks --
def check_nulls(df, columns):
    """Count null values per column.

    Args:
        df: PySpark DataFrame to check.
        columns: List of column names to check for nulls.

    Returns:
        Dict mapping column name to null count.
    """
    null_counts = {}
    for col_name in columns:
        if col_name in df.columns:
            count = df.filter(F.col(col_name).isNull()).count()
            null_counts[col_name] = count
    return null_counts


def check_duplicates(df, key_columns):
    """Count duplicate rows based on key columns.

    Args:
        df: PySpark DataFrame to check.
        key_columns: List of column names that form the unique key.

    Returns:
        Number of duplicate rows (total rows minus distinct key rows).
    """
    total = df.count()
    distinct = df.dropDuplicates(key_columns).count()
    return total - distinct


def check_freshness(df, timestamp_col, max_age_hours=48):
    """Check that the newest record is within the acceptable age threshold.

    Args:
        df: PySpark DataFrame to check.
        timestamp_col: Name of the timestamp column to evaluate.
        max_age_hours: Maximum acceptable age in hours (default 48).

    Returns:
        Dict with keys: newest (str), age_hours (float), is_fresh (bool).
        Returns default values if the timestamp column is missing or all null.
    """
    if timestamp_col not in df.columns:
        return {"newest": None, "age_hours": None, "is_fresh": False}

    result = df.agg(F.max(F.col(timestamp_col)).alias("max_ts")).first()
    max_ts = result["max_ts"]

    if max_ts is None:
        return {"newest": None, "age_hours": None, "is_fresh": False}

    # Handle both timestamp and string types
    if isinstance(max_ts, str):
        try:
            max_ts = datetime.fromisoformat(max_ts.replace("Z", "+00:00"))
        except ValueError:
            return {"newest": str(max_ts), "age_hours": None, "is_fresh": False}

    age_hours = (datetime.utcnow() - max_ts.replace(tzinfo=None)).total_seconds() / 3600
    is_fresh = age_hours <= max_age_hours

    return {
        "newest": str(max_ts),
        "age_hours": round(age_hours, 2),
        "is_fresh": is_fresh,
    }

# COMMAND ----------

# -- Unified Quality Metrics --
def compute_quality_metrics(df, source, batch_id, key_columns, required_columns,
                            timestamp_col=None):
    """Run all quality checks and return a unified metrics dict.

    Args:
        df: PySpark DataFrame to evaluate.
        source: Source identifier (e.g. "hackernews").
        batch_id: Batch identifier string.
        key_columns: Columns forming the unique key (for duplicate check).
        required_columns: Columns to check for nulls.
        timestamp_col: Optional timestamp column for freshness check.

    Returns:
        Dict with all quality metrics.
    """
    total_rows = df.count()
    null_counts = check_nulls(df, required_columns)
    duplicate_count = check_duplicates(df, key_columns)

    freshness = {"newest": None, "age_hours": None, "is_fresh": True}
    if timestamp_col:
        freshness = check_freshness(df, timestamp_col)

    return {
        "log_id": str(uuid.uuid4()),
        "source": source,
        "batch_id": batch_id,
        "total_rows": total_rows,
        "null_counts": null_counts,
        "duplicate_count": duplicate_count,
        "freshness_hours": freshness.get("age_hours"),
        "is_fresh": freshness.get("is_fresh", True),
        "checked_at": datetime.utcnow(),
    }

# COMMAND ----------

# -- Log Quality Metrics to Delta --
def log_quality_to_delta(spark, metrics):
    """Append a quality metrics record to the data quality log Delta table.

    Args:
        spark: The active SparkSession.
        metrics: Dict from compute_quality_metrics().
    """
    # Determine if quality gate passed before logging
    quality_passed = assert_quality_gate(metrics)

    row = (
        metrics["log_id"],
        metrics["source"],
        metrics["batch_id"],
        metrics["total_rows"],
        json.dumps(metrics["null_counts"]),
        metrics["duplicate_count"],
        metrics.get("freshness_hours"),
        metrics.get("is_fresh", True),
        quality_passed,
        metrics["checked_at"],
    )

    df = spark.createDataFrame([row], schema=DATA_QUALITY_LOG_SCHEMA)

    try:
        (
            df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(TABLE_NAMES["data_quality_log"])
        )
        logger.info(
            "Quality metrics logged for %s batch %s — %d rows, %d duplicates, passed=%s",
            metrics["source"], metrics["batch_id"], metrics["total_rows"],
            metrics["duplicate_count"], quality_passed,
        )
    except Exception as exc:
        logger.error("Failed to log quality metrics: %s", exc)
        raise

# COMMAND ----------

# -- Quality Gate --
# WHY: We warn but don't fail so the pipeline can complete and the quality
# log captures the issue for review. Hard failures on data quality would
# require manual intervention with no self-healing benefit — the data is
# already in Bronze and can be reprocessed.
def assert_quality_gate(metrics, max_null_pct=0.10, max_duplicate_pct=0.01):
    """Check if data passes quality thresholds.

    Logs warnings for each threshold breach but does NOT raise exceptions.

    Args:
        metrics: Dict from compute_quality_metrics().
        max_null_pct: Maximum acceptable null percentage per column (default 10%).
        max_duplicate_pct: Maximum acceptable duplicate percentage (default 1%).

    Returns:
        True if all checks pass, False if any threshold is breached.
    """
    total = metrics["total_rows"]
    if total == 0:
        logger.warning("Quality gate: 0 rows — nothing to check for %s", metrics["source"])
        return False

    passed = True

    # Check null rates
    for col_name, null_count in metrics["null_counts"].items():
        null_pct = null_count / total
        if null_pct > max_null_pct:
            logger.warning(
                "Quality gate BREACH: %s.%s has %.1f%% nulls (threshold: %.1f%%)",
                metrics["source"], col_name, null_pct * 100, max_null_pct * 100,
            )
            passed = False

    # Check duplicate rate
    dup_pct = metrics["duplicate_count"] / total
    if dup_pct > max_duplicate_pct:
        logger.warning(
            "Quality gate BREACH: %s has %.1f%% duplicates (threshold: %.1f%%)",
            metrics["source"], dup_pct * 100, max_duplicate_pct * 100,
        )
        passed = False

    # Check freshness
    if not metrics.get("is_fresh", True):
        logger.warning(
            "Quality gate BREACH: %s data is stale (%.1f hours old)",
            metrics["source"], metrics.get("freshness_hours", -1),
        )
        passed = False

    if passed:
        logger.info("Quality gate PASSED for %s (%d rows)", metrics["source"], total)

    return passed
