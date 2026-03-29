"""
Stratum — Central Configuration Module

Single source of truth for all configuration: API endpoints, database/table names,
Delta Lake properties, retry settings, and secret management.

This is a plain Python module (not a notebook) imported by all notebooks.
"""

import os
import logging

logger = logging.getLogger("stratum")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
API_ENDPOINTS = {
    "hackernews": "https://hacker-news.firebaseio.com/v0",
    "github": "https://api.github.com",
    "arxiv": "http://export.arxiv.org/api",
    "stackoverflow": "https://api.stackexchange.com/2.3",
}

# ---------------------------------------------------------------------------
# Database Names (Hive metastore — no Unity Catalog on Community Edition)
# ---------------------------------------------------------------------------
DATABASE_NAMES = {
    "bronze": "stratum_bronze",
    "silver": "stratum_silver",
    "gold": "stratum_gold",
}

# ---------------------------------------------------------------------------
# Table Names — fully qualified as database.table
# ---------------------------------------------------------------------------
TABLE_NAMES = {
    # Bronze
    "hackernews_raw": f"{DATABASE_NAMES['bronze']}.hackernews_raw",
    "github_raw": f"{DATABASE_NAMES['bronze']}.github_raw",
    "arxiv_raw": f"{DATABASE_NAMES['bronze']}.arxiv_raw",
    "stackoverflow_raw": f"{DATABASE_NAMES['bronze']}.stackoverflow_raw",
    "checkpoints": f"{DATABASE_NAMES['bronze']}._checkpoints",
    # Silver
    "tech_signals": f"{DATABASE_NAMES['silver']}.tech_signals",
    "data_quality_log": f"{DATABASE_NAMES['silver']}.data_quality_log",
    # Gold
    "tech_term_frequency": f"{DATABASE_NAMES['gold']}.tech_term_frequency",
    "trending_signals": f"{DATABASE_NAMES['gold']}.trending_signals",
    "source_summary": f"{DATABASE_NAMES['gold']}.source_summary",
    "tech_velocity": f"{DATABASE_NAMES['gold']}.tech_velocity",
    "llm_enriched": f"{DATABASE_NAMES['gold']}.llm_enriched",
}

# ---------------------------------------------------------------------------
# Delta Lake Table Properties
# WHY: autoOptimize prevents the small-file problem that arises from frequent
# appends on Community Edition where there is no scheduled OPTIMIZE job.
# ---------------------------------------------------------------------------
DELTA_PROPERTIES = {
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
}

# ---------------------------------------------------------------------------
# Spark Configuration
# WHY: Community Edition runs on a single node with limited cores. The default
# 200 shuffle partitions creates excessive small files and overhead. 8 is
# appropriate for our data volumes (thousands of rows, not millions).
# ---------------------------------------------------------------------------
SHUFFLE_PARTITIONS = 8

# ---------------------------------------------------------------------------
# Retry Configuration for API Calls
# ---------------------------------------------------------------------------
RETRY_CONFIG = {
    "max_retries": 3,
    "backoff_base": 2,
    "backoff_max": 30,
    "timeout": 30,
}

# ---------------------------------------------------------------------------
# LLM Enrichment Configuration
# ---------------------------------------------------------------------------
LLM_CONFIG = {
    "model": "gpt-3.5-turbo",
    "batch_size": 20,
    "max_tokens": 256,
    "temperature": 0.3,
}

# WHY: Fixed taxonomy keeps LLM output consistent and queryable. These are the
# categories the LLM must choose from — anything outside maps to "Other".
LLM_CATEGORIES = [
    "AI/ML",
    "Data Engineering",
    "Cloud",
    "DevOps",
    "Web",
    "Systems",
    "Security",
    "Other",
]

# ---------------------------------------------------------------------------
# GitHub Search Queries
# WHY: GitHub has no trending endpoint. We use the search API with
# created:>DATE sort:stars to find recently-created repos gaining traction.
# ---------------------------------------------------------------------------
GITHUB_SEARCH_TOPICS = [
    "machine learning",
    "data engineering",
    "LLM",
    "generative AI",
    "databricks",
    "spark",
]

# ---------------------------------------------------------------------------
# Stack Overflow Tags to Track
# ---------------------------------------------------------------------------
STACKOVERFLOW_TAGS = [
    "python",
    "apache-spark",
    "databricks",
    "machine-learning",
    "llm",
    "data-engineering",
    "deep-learning",
    "pytorch",
]

# ---------------------------------------------------------------------------
# arXiv Categories
# ---------------------------------------------------------------------------
ARXIV_CATEGORIES = [
    "cs.AI",
    "cs.LG",
    "cs.CL",
]


# ---------------------------------------------------------------------------
# Secret Management
# ---------------------------------------------------------------------------
def get_secret(key):
    """Retrieve a secret, trying Databricks secrets first then environment variables.

    Args:
        key: The secret key name (e.g. 'OPENAI_API_KEY', 'GITHUB_TOKEN').

    Returns:
        The secret string value.

    Raises:
        ValueError: If the secret is not found in either location.
    """
    # WHY: Community Edition has limited secrets support and dbutils may not
    # be available in local/CI testing. The fallback to os.environ keeps the
    # code portable without compromising security on Databricks.
    try:
        # dbutils is injected by the Databricks runtime — not importable directly
        return dbutils.secrets.get(scope="stratum", key=key)  # noqa: F821
    except Exception:
        value = os.environ.get(key)
        if value is None:
            raise ValueError(
                f"Secret '{key}' not found in Databricks secrets (scope=stratum) "
                f"or environment variable '{key}'."
            )
        return value


def apply_spark_config(spark):
    """Apply standard Spark configuration to the active session.

    Args:
        spark: The active SparkSession.

    Sets:
        - spark.sql.shuffle.partitions = 8 (optimised for single-node CE)
        - spark.sql.session.timeZone = UTC (all timestamps normalised to UTC)
    """
    spark.conf.set("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    logger.info(
        "Spark config applied: shuffle.partitions=%d, timeZone=UTC",
        SHUFFLE_PARTITIONS,
    )


def setup_logging(level=logging.INFO):
    """Configure the stratum logger with a consistent format.

    Args:
        level: Logging level (default INFO).

    Returns:
        The configured logger instance.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger = logging.getLogger("stratum")
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(level)
    return root_logger
