"""
Stratum — Delta Lake Utilities

Reusable helpers for Delta Lake operations: database/table creation, bronze append,
MERGE upsert, OPTIMIZE, and checkpoint management for the simulated Auto Loader pattern.
"""

import logging
import uuid
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from delta.tables import DeltaTable

from config import DATABASE_NAMES, TABLE_NAMES, DELTA_PROPERTIES

logger = logging.getLogger("stratum.delta")


# ---------------------------------------------------------------------------
# Database & Table Creation
# ---------------------------------------------------------------------------
def create_database_if_not_exists(spark, db_name):
    """Create a Hive metastore database if it does not already exist.

    Args:
        spark: The active SparkSession.
        db_name: Database name to create.
    """
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    logger.info("Ensured database exists: %s", db_name)


def create_all_databases(spark):
    """Create all Stratum databases (bronze, silver, gold).

    Args:
        spark: The active SparkSession.
    """
    for db_name in DATABASE_NAMES.values():
        create_database_if_not_exists(spark, db_name)


def create_delta_table(spark, table_name, schema, partition_cols=None, properties=None):
    """Create a managed Delta table if it does not already exist.

    Applies Stratum default Delta properties (autoOptimize) unless overridden.

    Args:
        spark: The active SparkSession.
        table_name: Fully qualified table name (e.g. "stratum_bronze.hackernews_raw").
        schema: PySpark StructType defining the table schema.
        partition_cols: Optional list of column names to partition by.
        properties: Optional dict of Delta table properties. Merged with defaults.
    """
    merged_props = {**DELTA_PROPERTIES, **(properties or {})}

    if _table_exists(spark, table_name):
        logger.info("Table already exists: %s", table_name)
        return

    builder = (
        DeltaTable.createIfNotExists(spark)
        .tableName(table_name)
    )

    for field in schema.fields:
        builder = builder.addColumn(field.name, field.dataType, nullable=field.nullable)

    if partition_cols:
        builder = builder.partitionedBy(*partition_cols)

    for key, value in merged_props.items():
        builder = builder.property(key, value)

    try:
        builder.execute()
        logger.info("Created Delta table: %s", table_name)
    except Exception as exc:
        logger.error("Failed to create Delta table %s: %s", table_name, exc)
        raise


def _table_exists(spark, table_name):
    """Check if a table exists in the Hive metastore.

    Args:
        spark: The active SparkSession.
        table_name: Fully qualified table name.

    Returns:
        True if the table exists, False otherwise.
    """
    db, table = table_name.split(".", 1)
    tables = [t.name for t in spark.catalog.listTables(db)]
    return table in tables


# ---------------------------------------------------------------------------
# Bronze Write — Append Only
# WHY: Bronze is an immutable audit log of exactly what the APIs returned.
# Append-only preserves this property. Deduplication is Silver's job.
# ---------------------------------------------------------------------------
def write_bronze(spark, df, table_name, batch_id, source_name):
    """Append a DataFrame to a bronze Delta table with ingestion metadata.

    Adds three metadata columns: _ingested_at, _source, _batch_id.

    Args:
        spark: The active SparkSession.
        df: DataFrame to write (without metadata columns).
        table_name: Target bronze table (fully qualified).
        batch_id: Unique batch identifier string.
        source_name: Source identifier (e.g. "hackernews").

    Returns:
        Number of rows written.

    Raises:
        Exception: If the write fails (logged before re-raising).
    """
    df_with_meta = (
        df
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source", F.lit(source_name))
        .withColumn("_batch_id", F.lit(batch_id))
    )

    row_count = df_with_meta.count()

    try:
        (
            df_with_meta.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(table_name)
        )
        logger.info("Wrote %d rows to %s (batch: %s)", row_count, table_name, batch_id)
        return row_count
    except Exception as exc:
        logger.error("Failed to write to %s: %s", table_name, exc)
        raise


# ---------------------------------------------------------------------------
# MERGE Upsert — for Silver and Gold
# WHY: MERGE guarantees idempotency — re-running a batch won't create
# duplicates. This is essential for a resumable pipeline.
# ---------------------------------------------------------------------------
def upsert_to_table(spark, source_df, target_table, merge_keys, update_columns=None):
    """MERGE source DataFrame into target Delta table.

    When matched on merge_keys: updates specified columns (or all non-key columns).
    When not matched: inserts all columns.

    Args:
        spark: The active SparkSession.
        source_df: DataFrame with new/updated rows.
        target_table: Fully qualified target table name.
        merge_keys: List of column names to join on.
        update_columns: Optional list of columns to update on match.
            If None, updates all columns not in merge_keys.

    Returns:
        Dict with row count of the source DataFrame.
    """
    source_count = source_df.count()
    if source_count == 0:
        logger.info("No rows to upsert into %s — skipping", target_table)
        return {"source_rows": 0}

    # Build the merge condition
    merge_condition = " AND ".join(
        [f"target.{key} = source.{key}" for key in merge_keys]
    )

    if not _table_exists(spark, target_table):
        # First write — no table to merge into yet
        try:
            (
                source_df.write
                .format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(target_table)
            )
            # Apply Delta properties after creation
            for prop_key, prop_val in DELTA_PROPERTIES.items():
                spark.sql(f"ALTER TABLE {target_table} SET TBLPROPERTIES ('{prop_key}' = '{prop_val}')")
            logger.info("Created %s with %d rows (first upsert)", target_table, source_count)
            return {"source_rows": source_count}
        except Exception as exc:
            logger.error("Failed initial write to %s: %s", target_table, exc)
            raise

    target_dt = DeltaTable.forName(spark, target_table)

    # Determine columns to update
    if update_columns is None:
        update_columns = [c for c in source_df.columns if c not in merge_keys]

    update_set = {col: f"source.{col}" for col in update_columns}
    insert_set = {col: f"source.{col}" for col in source_df.columns}

    try:
        (
            target_dt.alias("target")
            .merge(source_df.alias("source"), merge_condition)
            .whenMatchedUpdate(set=update_set)
            .whenNotMatchedInsert(values=insert_set)
            .execute()
        )
        logger.info("Upserted %d source rows into %s", source_count, target_table)
        return {"source_rows": source_count}
    except Exception as exc:
        logger.error("Failed to upsert into %s: %s", target_table, exc)
        raise


# ---------------------------------------------------------------------------
# OPTIMIZE + ZORDER
# ---------------------------------------------------------------------------
def optimize_table(spark, table_name, zorder_cols=None):
    """Run OPTIMIZE on a Delta table, optionally with ZORDER BY.

    Args:
        spark: The active SparkSession.
        table_name: Fully qualified table name.
        zorder_cols: Optional list of columns to ZORDER by.
    """
    try:
        if zorder_cols:
            zorder_clause = ", ".join(zorder_cols)
            spark.sql(f"OPTIMIZE {table_name} ZORDER BY ({zorder_clause})")
            logger.info("Optimized %s with ZORDER BY (%s)", table_name, zorder_clause)
        else:
            spark.sql(f"OPTIMIZE {table_name}")
            logger.info("Optimized %s", table_name)
    except Exception as exc:
        logger.error("Failed to optimize %s: %s", table_name, exc)
        raise


# ---------------------------------------------------------------------------
# Checkpoint Management — Simulated Auto Loader
# WHY: Community Edition has no Auto Loader (no structured streaming with
# cloudFiles). We simulate the three key guarantees — incremental processing,
# exactly-once semantics, and checkpoint recovery — using a Delta table.
# A Delta table (vs filesystem marker) supports atomic MERGE, is queryable
# for debugging, and lives in the same metastore as all other tables.
# ---------------------------------------------------------------------------
CHECKPOINT_SCHEMA = StructType([
    StructField("source", StringType(), False),
    StructField("last_id", StringType(), True),
    StructField("last_timestamp", StringType(), True),
    StructField("last_batch_id", StringType(), True),
    StructField("updated_at", TimestampType(), True),
])


def _ensure_checkpoint_table(spark):
    """Create the checkpoint table if it does not exist.

    Args:
        spark: The active SparkSession.
    """
    create_database_if_not_exists(spark, DATABASE_NAMES["bronze"])
    create_delta_table(spark, TABLE_NAMES["checkpoints"], CHECKPOINT_SCHEMA)


def get_checkpoint(spark, source):
    """Read the last checkpoint for a data source.

    Args:
        spark: The active SparkSession.
        source: Source identifier (e.g. "hackernews", "github").

    Returns:
        Dict with keys last_id, last_timestamp, last_batch_id, updated_at.
        Empty dict if no checkpoint exists for this source.
    """
    _ensure_checkpoint_table(spark)

    try:
        row = (
            spark.table(TABLE_NAMES["checkpoints"])
            .filter(F.col("source") == source)
            .first()
        )
    except Exception:
        return {}

    if row is None:
        return {}

    return {
        "last_id": row["last_id"],
        "last_timestamp": row["last_timestamp"],
        "last_batch_id": row["last_batch_id"],
        "updated_at": row["updated_at"],
    }


def save_checkpoint(spark, source, last_id=None, last_timestamp=None, batch_id=None):
    """Save or update the checkpoint for a data source.

    Uses MERGE to atomically upsert — safe even on concurrent runs.

    Args:
        spark: The active SparkSession.
        source: Source identifier.
        last_id: The high-water mark ID (e.g. max HN story ID).
        last_timestamp: The high-water mark timestamp (ISO string).
        batch_id: The batch ID associated with this checkpoint.
    """
    _ensure_checkpoint_table(spark)

    checkpoint_df = spark.createDataFrame(
        [(source, last_id, last_timestamp, batch_id, datetime.utcnow())],
        schema=CHECKPOINT_SCHEMA,
    )

    upsert_to_table(
        spark,
        source_df=checkpoint_df,
        target_table=TABLE_NAMES["checkpoints"],
        merge_keys=["source"],
    )
    logger.info(
        "Checkpoint saved for %s — last_id=%s, last_timestamp=%s, batch=%s",
        source, last_id, last_timestamp, batch_id,
    )


# ---------------------------------------------------------------------------
# Batch ID Generation
# ---------------------------------------------------------------------------
def generate_batch_id(source):
    """Generate a unique batch identifier.

    Format: {source}_{YYYYMMDD_HHMMSS}_{uuid4_short}

    Args:
        source: Source identifier (e.g. "hackernews").

    Returns:
        Batch ID string.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{source}_{timestamp}_{short_uuid}"
