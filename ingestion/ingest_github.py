# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 — Ingest GitHub
# MAGIC
# MAGIC **What this notebook does:** Searches for trending repositories using the GitHub
# MAGIC Search API (`/search/repositories?q=...&sort=stars`) and writes raw results to
# MAGIC `stratum_bronze.github_raw` as a Delta table.
# MAGIC
# MAGIC **Inputs:** GitHub public REST API (search endpoint, no auth required within rate limits)
# MAGIC
# MAGIC **Outputs:** `stratum_bronze.github_raw` Delta table
# MAGIC
# MAGIC **Dependencies:** `config.py`, `utils/api_utils.py`, `utils/delta_utils.py`
# MAGIC
# MAGIC **Design note:** GitHub has no official trending endpoint. We use the search API with
# MAGIC `created:>DATE sort:stars` queries to find recently-created repos gaining traction.
# MAGIC Unauthenticated rate limit is 10 requests/minute — the pagination helper enforces this.

# COMMAND ----------

import json
import logging
from datetime import datetime, timedelta

from pyspark.sql.types import (
    StructType, StructField, LongType, StringType, IntegerType, DoubleType,
)

from config import (
    TABLE_NAMES, GITHUB_SEARCH_TOPICS, apply_spark_config, setup_logging,
)
from utils.api_utils import paginate_github
from utils.delta_utils import (
    create_all_databases, create_delta_table, write_bronze,
    get_checkpoint, save_checkpoint, generate_batch_id,
)
from utils.quality_utils import compute_quality_metrics, log_quality_to_delta

logger = setup_logging()

# COMMAND ----------

# -- Widget definitions --
dbutils.widgets.text(
    "search_queries",
    ",".join(GITHUB_SEARCH_TOPICS),
    "Comma-separated search terms",
)
dbutils.widgets.text("max_pages", "3", "Max pages per search query")
dbutils.widgets.text("lookback_days", "7", "Days to look back for created repos")
dbutils.widgets.dropdown("run_mode", "incremental", ["incremental", "full"], "Run mode")

SEARCH_QUERIES = [q.strip() for q in dbutils.widgets.get("search_queries").split(",")]
MAX_PAGES = int(dbutils.widgets.get("max_pages"))
LOOKBACK_DAYS = int(dbutils.widgets.get("lookback_days"))
RUN_MODE = dbutils.widgets.get("run_mode")

logger.info(
    "GitHub ingestion — queries=%s, max_pages=%d, lookback=%dd, run_mode=%s",
    SEARCH_QUERIES, MAX_PAGES, LOOKBACK_DAYS, RUN_MODE,
)

# COMMAND ----------

apply_spark_config(spark)
create_all_databases(spark)

# COMMAND ----------

# -- Bronze table schema --
GITHUB_BRONZE_SCHEMA = StructType([
    StructField("id", LongType(), False),
    StructField("name", StringType(), True),
    StructField("full_name", StringType(), True),
    StructField("description", StringType(), True),
    StructField("html_url", StringType(), True),
    StructField("stargazers_count", IntegerType(), True),
    StructField("forks_count", IntegerType(), True),
    StructField("language", StringType(), True),
    StructField("created_at", StringType(), True),
    StructField("updated_at", StringType(), True),
    StructField("pushed_at", StringType(), True),
    StructField("owner_login", StringType(), True),
    StructField("topics", StringType(), True),
    StructField("open_issues_count", IntegerType(), True),
    StructField("score", DoubleType(), True),
    StructField("_raw_json", StringType(), True),
])

create_delta_table(spark, TABLE_NAMES["github_raw"], GITHUB_BRONZE_SCHEMA)

# COMMAND ----------

def build_search_query(topic, created_after_date):
    """Build a GitHub search query string for a topic with date filter.

    Args:
        topic: Search topic string (e.g. "machine learning").
        created_after_date: ISO date string (e.g. "2026-03-22"). Only repos
            created after this date are returned.

    Returns:
        Formatted query string for GitHub search API.
    """
    # WHY: Quoting the topic ensures multi-word queries like "machine learning"
    # are treated as a phrase by the GitHub search API.
    return f'"{topic}" created:>{created_after_date} sort:stars'


def fetch_trending_repos(queries, max_pages, lookback_days):
    """Fetch trending repos across multiple search queries, deduplicated.

    Args:
        queries: List of search topic strings.
        max_pages: Maximum pages to fetch per query.
        lookback_days: Number of days to look back for repo creation.

    Returns:
        List of unique repo dicts (deduplicated by full_name).
    """
    created_after = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    all_repos = {}

    for topic in queries:
        query = build_search_query(topic, created_after)
        logger.info("GitHub search: %s", query)

        repos = paginate_github(query, max_pages=max_pages, per_page=30)

        for repo in repos:
            full_name = repo.get("full_name", "")
            if full_name and full_name not in all_repos:
                all_repos[full_name] = repo

        logger.info(
            "Topic '%s': %d repos fetched, %d unique total",
            topic, len(repos), len(all_repos),
        )

    return list(all_repos.values())

# COMMAND ----------

def build_bronze_df(repos, batch_id):
    """Convert a list of repo dicts into a PySpark DataFrame matching bronze schema.

    Args:
        repos: List of dicts from fetch_trending_repos().
        batch_id: The batch identifier.

    Returns:
        PySpark DataFrame with columns matching GITHUB_BRONZE_SCHEMA.
    """
    rows = []
    for r in repos:
        owner = r.get("owner", {}) or {}
        rows.append((
            int(r.get("id", 0)),
            r.get("name"),
            r.get("full_name"),
            r.get("description"),
            r.get("html_url"),
            r.get("stargazers_count"),
            r.get("forks_count"),
            r.get("language"),
            r.get("created_at"),
            r.get("updated_at"),
            r.get("pushed_at"),
            owner.get("login"),
            json.dumps(r.get("topics", [])),
            r.get("open_issues_count"),
            float(r.get("score", 0) or 0),
            json.dumps(r),
        ))
    return spark.createDataFrame(rows, schema=GITHUB_BRONZE_SCHEMA)

# COMMAND ----------

# -- Main ingestion logic --
batch_id = generate_batch_id("github")
logger.info("Starting GitHub ingestion — batch_id=%s", batch_id)

repos = fetch_trending_repos(SEARCH_QUERIES, MAX_PAGES, LOOKBACK_DAYS)

if not repos:
    logger.info("No GitHub repos found — exiting")
    dbutils.notebook.exit("NO_NEW_DATA")

df = build_bronze_df(repos, batch_id)

# COMMAND ----------

# -- Data quality check --
quality_metrics = compute_quality_metrics(
    df=df,
    source="github",
    batch_id=batch_id,
    key_columns=["full_name"],
    required_columns=["id", "full_name", "html_url", "stargazers_count"],
)
log_quality_to_delta(spark, quality_metrics)

# COMMAND ----------

# -- Write to Bronze --
rows_written = write_bronze(spark, df, TABLE_NAMES["github_raw"], batch_id, "github")

# -- Update checkpoint with most recent pushed_at timestamp --
max_pushed = max(
    (r.get("pushed_at", "") for r in repos),
    default="",
)
save_checkpoint(spark, "github", last_timestamp=max_pushed, batch_id=batch_id)

logger.info(
    "GitHub ingestion complete — %d rows written, latest pushed_at=%s",
    rows_written, max_pushed,
)

# COMMAND ----------

# -- Verification --
display(spark.table(TABLE_NAMES["github_raw"]).filter(f"_batch_id = '{batch_id}'").limit(5))
print(f"Total rows in {TABLE_NAMES['github_raw']}: {spark.table(TABLE_NAMES['github_raw']).count()}")

# COMMAND ----------

dbutils.notebook.exit(f"SUCCESS: {rows_written} rows ingested")
