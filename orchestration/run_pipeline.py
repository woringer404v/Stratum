# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 09 — Run Pipeline (Orchestration)
# MAGIC
# MAGIC **What this notebook does:** Master orchestrator that runs all Stratum pipeline steps
# MAGIC in sequence with error handling, timing, and a summary report. This is the single
# MAGIC entry point — run this notebook to execute the entire pipeline end-to-end.
# MAGIC
# MAGIC **Inputs:** None (reads widgets for configuration)
# MAGIC
# MAGIC **Outputs:** Executes all upstream notebooks; displays pipeline summary
# MAGIC
# MAGIC **Dependencies:** All other notebooks must be importable via relative paths
# MAGIC
# MAGIC **Design note:** `dbutils.notebook.run()` is synchronous on Community Edition.
# MAGIC No parallel execution is possible without the Jobs API. Steps run sequentially
# MAGIC with per-step error handling — a failure in one step is logged but does not
# MAGIC halt the pipeline (unless it's a critical dependency).

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

import time
import logging
from datetime import datetime

logger = setup_logging()

# COMMAND ----------

# -- Widget definitions --
dbutils.widgets.dropdown("run_mode", "incremental", ["incremental", "full"], "Run mode")
dbutils.widgets.dropdown("skip_llm", "false", ["true", "false"], "Skip LLM enrichment")
dbutils.widgets.dropdown("llm_provider", LLM_CONFIG["provider"], list(LLM_PROVIDERS.keys()), "LLM provider")
dbutils.widgets.text("llm_api_key", "", "LLM API key (paste here)")
dbutils.widgets.text("batch_size", "200", "HN batch size")
dbutils.widgets.text("max_signals", "50", "Max signals for LLM enrichment")

RUN_MODE = dbutils.widgets.get("run_mode")
SKIP_LLM = dbutils.widgets.get("skip_llm") == "true"
LLM_PROVIDER = dbutils.widgets.get("llm_provider")
LLM_API_KEY = dbutils.widgets.get("llm_api_key").strip()
BATCH_SIZE = dbutils.widgets.get("batch_size")
MAX_SIGNALS = dbutils.widgets.get("max_signals")

logger.info("Pipeline starting — run_mode=%s, skip_llm=%s, llm_provider=%s", RUN_MODE, SKIP_LLM, LLM_PROVIDER)

# COMMAND ----------

apply_spark_config(spark)

# COMMAND ----------

# -- Set and validate LLM API key --
# WHY: dbutils.notebook.run() creates a new Python process. Environment
# variables set in this notebook don't carry over. We use Spark config
# as a transport mechanism — it persists across notebook.run() calls.
import os, requests

if not SKIP_LLM:
    if not LLM_API_KEY:
        logger.warning("No LLM API key provided — setting skip_llm=true")
        SKIP_LLM = True
    else:
        # Store key in env var AND Spark conf so child notebooks can access it
        provider_cfg = LLM_PROVIDERS[LLM_PROVIDER]
        env_key = provider_cfg["env_key"]
        os.environ[env_key] = LLM_API_KEY
        spark.conf.set(f"stratum.secret.{env_key}", LLM_API_KEY)

        # Quick validation — send a tiny test request
        logger.info("Validating %s API key...", LLM_PROVIDER)
        try:
            if LLM_PROVIDER == "openai":
                r = requests.post(
                    provider_cfg["url"],
                    headers=provider_cfg["auth_header"](LLM_API_KEY),
                    json={"model": provider_cfg["model"], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                    timeout=15,
                )
            elif LLM_PROVIDER == "anthropic":
                r = requests.post(
                    provider_cfg["url"],
                    headers=provider_cfg["auth_header"](LLM_API_KEY),
                    json={"model": provider_cfg["model"], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                    timeout=15,
                )
            elif LLM_PROVIDER == "gemini":
                url = provider_cfg["url"].replace("{model}", provider_cfg["model"]) + f"?key={LLM_API_KEY}"
                r = requests.post(
                    url,
                    headers=provider_cfg["auth_header"](LLM_API_KEY),
                    json={"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 5}},
                    timeout=15,
                )

            if r.status_code == 200:
                logger.info("API key validated — %s is working", LLM_PROVIDER)
            else:
                logger.error("API key validation FAILED — %s returned %d: %s", LLM_PROVIDER, r.status_code, r.text[:200])
                SKIP_LLM = True
                logger.warning("Disabling LLM enrichment due to invalid API key")
        except Exception as exc:
            logger.error("API key validation error: %s", exc)
            SKIP_LLM = True

# COMMAND ----------

def run_notebook(path, timeout_seconds, params=None):
    """Execute a notebook via dbutils.notebook.run with error handling.

    Args:
        path: Relative path to the notebook (e.g. "../ingestion/ingest_hackernews").
        timeout_seconds: Maximum execution time in seconds.
        params: Optional dict of widget parameters to pass.

    Returns:
        Tuple of (status: str, result: str, duration_seconds: float).
    """
    params = params or {}
    start = time.time()

    try:
        result = dbutils.notebook.run(path, timeout_seconds, params)
        duration = time.time() - start
        logger.info("PASS — %s completed in %.1fs: %s", path, duration, result)
        return ("SUCCESS", result, duration)
    except Exception as exc:
        duration = time.time() - start
        error_msg = str(exc)[:200]
        logger.error("FAIL — %s failed after %.1fs: %s", path, duration, error_msg)
        return ("FAILED", error_msg, duration)

# COMMAND ----------

# -- Pipeline step definitions --
# WHY: dbutils.notebook.run is synchronous on Community Edition — no parallel
# execution possible without Jobs API. Steps run in dependency order.
PIPELINE_STEPS = [
    {
        "name": "Ingest Hacker News",
        "path": "../ingestion/ingest_hackernews",
        "timeout": 600,
        "params": {"batch_size": BATCH_SIZE, "run_mode": RUN_MODE},
        "critical": False,
    },
    {
        "name": "Ingest GitHub",
        "path": "../ingestion/ingest_github",
        "timeout": 900,
        "params": {"run_mode": RUN_MODE},
        "critical": False,
    },
    {
        "name": "Ingest arXiv",
        "path": "../ingestion/ingest_arxiv",
        "timeout": 600,
        "params": {"run_mode": RUN_MODE},
        "critical": False,
    },
    {
        "name": "Ingest Stack Overflow",
        "path": "../ingestion/ingest_stackoverflow",
        "timeout": 600,
        "params": {"run_mode": RUN_MODE},
        "critical": False,
    },
    {
        "name": "Bronze to Silver",
        "path": "../silver/bronze_to_silver",
        "timeout": 900,
        "params": {"reprocess": "true" if RUN_MODE == "full" else "false"},
        "critical": True,
    },
    {
        "name": "Silver to Gold",
        "path": "../gold/silver_to_gold",
        "timeout": 900,
        "params": {},
        "critical": True,
    },
    {
        "name": "LLM Enrichment",
        "path": "../gold/llm_enrichment",
        "timeout": 1800,
        "params": {"max_signals": MAX_SIGNALS, "provider": LLM_PROVIDER},
        "critical": False,
        "skip_condition": SKIP_LLM,
    },
    {
        "name": "Analytics Queries",
        "path": "../analytics/analytics_queries",
        "timeout": 300,
        "params": {},
        "critical": False,
    },
]

# COMMAND ----------

# -- Execute pipeline --
pipeline_start = time.time()
results = []
has_critical_failure = False

for step in PIPELINE_STEPS:
    step_name = step["name"]

    # Check skip condition
    if step.get("skip_condition", False):
        logger.info("SKIP — %s (skip condition met)", step_name)
        results.append({
            "step": step_name,
            "status": "SKIPPED",
            "result": "Skip condition met",
            "duration": 0.0,
        })
        continue

    # Skip non-critical steps after a critical failure
    if has_critical_failure and not step.get("critical", False):
        logger.warning("SKIP — %s (previous critical failure)", step_name)
        results.append({
            "step": step_name,
            "status": "SKIPPED",
            "result": "Skipped due to critical failure",
            "duration": 0.0,
        })
        continue

    # Run the step
    status, result, duration = run_notebook(
        step["path"], step["timeout"], step.get("params"),
    )

    results.append({
        "step": step_name,
        "status": status,
        "result": result,
        "duration": duration,
    })

    if status == "FAILED" and step.get("critical", False):
        has_critical_failure = True
        logger.error("Critical step failed: %s — downstream steps may be skipped", step_name)

pipeline_duration = time.time() - pipeline_start
logger.info("Pipeline complete in %.1fs", pipeline_duration)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Summary

# COMMAND ----------

# -- Display pipeline results --
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

summary_schema = StructType([
    StructField("Step", StringType(), False),
    StructField("Status", StringType(), False),
    StructField("Result", StringType(), True),
    StructField("Duration_sec", DoubleType(), False),
])

summary_rows = [
    (r["step"], r["status"], str(r["result"])[:100], round(r["duration"], 1))
    for r in results
]

summary_df = spark.createDataFrame(summary_rows, schema=summary_schema)
display(summary_df)

# COMMAND ----------

# -- Pipeline stats --
total_success = sum(1 for r in results if r["status"] == "SUCCESS")
total_failed = sum(1 for r in results if r["status"] == "FAILED")
total_skipped = sum(1 for r in results if r["status"] == "SKIPPED")

print(f"""
Pipeline Execution Summary
{'='*50}
Total steps:    {len(results)}
Successful:     {total_success}
Failed:         {total_failed}
Skipped:        {total_skipped}
Total duration: {pipeline_duration:.1f}s ({pipeline_duration/60:.1f}m)
{'='*50}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table Row Counts — Final Validation

# COMMAND ----------

# -- Row counts for all tables --
table_counts = []
for key, table_name in TABLE_NAMES.items():
    try:
        count = spark.table(table_name).count()
        table_counts.append((table_name, count))
    except Exception:
        table_counts.append((table_name, -1))

counts_df = spark.createDataFrame(
    table_counts,
    schema=StructType([
        StructField("Table", StringType()),
        StructField("Row_Count", StringType()),
    ]),
)
display(counts_df)

# COMMAND ----------

# -- Final status --
if total_failed > 0:
    exit_msg = f"COMPLETED_WITH_ERRORS: {total_success}/{len(results)} steps succeeded"
else:
    exit_msg = f"SUCCESS: All {total_success} steps completed in {pipeline_duration:.0f}s"

logger.info(exit_msg)
dbutils.notebook.exit(exit_msg)
