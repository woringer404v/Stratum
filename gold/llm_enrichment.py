# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 07 — LLM Enrichment
# MAGIC
# MAGIC **What this notebook does:** Enriches Silver signals with LLM-powered analysis:
# MAGIC category tagging, one-sentence summary, sentiment classification, and emerging-tech
# MAGIC detection. Results are written to `stratum_gold.llm_enriched`.
# MAGIC
# MAGIC **Inputs:** `stratum_silver.tech_signals` (LEFT ANTI JOIN with existing enrichments)
# MAGIC
# MAGIC **Outputs:** `stratum_gold.llm_enriched` Delta table
# MAGIC
# MAGIC **Dependencies:** `config.py`, `utils/api_utils.py`, `utils/delta_utils.py`
# MAGIC
# MAGIC **Architecture:** Driver-side batch loop (not a distributed UDF) because Community
# MAGIC Edition is single-node and LLM calls are I/O-bound. Each batch is upserted immediately
# MAGIC after completion for fault tolerance — if the notebook crashes at batch 5 of 10,
# MAGIC re-running skips the first 4 batches via LEFT ANTI JOIN.
# MAGIC
# MAGIC **Production alternative:** A Pandas UDF approach is documented below (CMD 10) for
# MAGIC reference — it would distribute LLM calls across executors on a multi-node cluster.

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# MAGIC %run ../utils/api_utils

# COMMAND ----------

# MAGIC %run ../utils/delta_utils

# COMMAND ----------

import json
import logging
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType, TimestampType,
)

logger = setup_logging()

# COMMAND ----------

# -- Widget definitions --
dbutils.widgets.text("batch_size", str(LLM_CONFIG["batch_size"]), "Signals per LLM batch")
dbutils.widgets.text("max_signals", "200", "Max signals to enrich per run")
dbutils.widgets.text("model", LLM_CONFIG["model"], "OpenAI model to use")

BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
MAX_SIGNALS = int(dbutils.widgets.get("max_signals"))
MODEL = dbutils.widgets.get("model")

logger.info(
    "LLM enrichment — batch_size=%d, max_signals=%d, model=%s",
    BATCH_SIZE, MAX_SIGNALS, MODEL,
)

# COMMAND ----------

apply_spark_config(spark)
create_all_databases(spark)

# COMMAND ----------

# -- LLM Enriched table schema --
LLM_ENRICHED_SCHEMA = StructType([
    StructField("signal_id", StringType(), False),
    StructField("category", StringType(), True),
    StructField("summary", StringType(), True),
    StructField("sentiment", StringType(), True),
    StructField("is_emerging", BooleanType(), True),
    StructField("model_used", StringType(), True),
    StructField("enriched_at", TimestampType(), True),
])

# COMMAND ----------

# -- System prompt for LLM --
CATEGORIES_STR = ", ".join(LLM_CATEGORIES)

SYSTEM_PROMPT = f"""You are a tech industry analyst. For each tech signal, provide a JSON response with exactly these fields:
- "category": one of [{CATEGORIES_STR}]
- "summary": a single concise sentence summarizing the signal
- "sentiment": one of ["positive", "neutral", "negative"]
- "is_emerging": boolean — true if this topic is new/emerging in the past week, false if established

Respond ONLY with valid JSON. No markdown, no explanation."""


def build_prompt(title, body, source, tags):
    """Construct the user prompt for a single signal.

    Args:
        title: Signal title.
        body: Signal body text (truncated to 500 chars).
        source: Data source name.
        tags: List of tags or comma-separated string.

    Returns:
        Formatted prompt string.
    """
    # WHY: Truncating body to 500 chars controls token usage and cost.
    # Titles and tags carry most of the classification signal anyway.
    body_truncated = (body or "")[:500]
    tags_str = tags if isinstance(tags, str) else ", ".join(tags or [])

    return f"""Analyse this tech signal:
Source: {source}
Title: {title}
Tags: {tags_str}
Body: {body_truncated}

Respond with JSON: {{"category": "...", "summary": "...", "sentiment": "...", "is_emerging": true/false}}"""

# COMMAND ----------

def call_llm_single(title, body, source, tags, api_key, model):
    """Call the OpenAI ChatCompletion API for a single signal.

    Args:
        title: Signal title.
        body: Signal body.
        source: Data source.
        tags: Signal tags.
        api_key: OpenAI API key.
        model: Model identifier.

    Returns:
        Dict with category, summary, sentiment, is_emerging, or defaults on failure.
    """
    prompt = build_prompt(title, body, source, tags)

    try:
        response = post_with_retry(
            url="https://api.openai.com/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": LLM_CONFIG["max_tokens"],
                "temperature": LLM_CONFIG["temperature"],
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse the JSON response
        parsed = json.loads(content)

        # Validate category against taxonomy
        category = parsed.get("category", "Other")
        if category not in LLM_CATEGORIES:
            category = "Other"

        # Validate sentiment
        sentiment = parsed.get("sentiment", "neutral")
        if sentiment not in ("positive", "neutral", "negative"):
            sentiment = "neutral"

        return {
            "category": category,
            "summary": parsed.get("summary", ""),
            "sentiment": sentiment,
            "is_emerging": bool(parsed.get("is_emerging", False)),
        }

    except json.JSONDecodeError as exc:
        logger.warning("LLM response not valid JSON: %s", exc)
        return _default_enrichment()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return _default_enrichment()


def _default_enrichment():
    """Return default enrichment values when LLM call fails.

    Returns:
        Dict with safe default values.
    """
    # WHY: Returning defaults instead of raising ensures the signal_id gets
    # written to llm_enriched, preventing re-processing on the next run.
    # The "error" category makes failures easy to find and re-enrich later.
    return {
        "category": "error",
        "summary": "Enrichment failed — retry later",
        "sentiment": "neutral",
        "is_emerging": False,
    }

# COMMAND ----------

def call_llm_batch(rows, api_key, model):
    """Process a batch of signals through the LLM.

    Args:
        rows: List of Row objects with signal_id, title, body, source, tags.
        api_key: OpenAI API key.
        model: Model identifier.

    Returns:
        List of enrichment result dicts with signal_id attached.
    """
    results = []
    for row in rows:
        enrichment = call_llm_single(
            title=row["title"],
            body=row["body"],
            source=row["source"],
            tags=row["tags"],
            api_key=api_key,
            model=model,
        )
        enrichment["signal_id"] = row["signal_id"]
        enrichment["model_used"] = model
        enrichment["enriched_at"] = datetime.utcnow()
        results.append(enrichment)

    return results

# COMMAND ----------

# -- Find signals that need enrichment --
# WHY: LEFT ANTI JOIN ensures we never re-process already enriched rows,
# making the notebook idempotent and resumable.
silver_df = spark.table(TABLE_NAMES["tech_signals"])

try:
    enriched_df = spark.table(TABLE_NAMES["llm_enriched"])
    candidates = (
        silver_df.alias("s")
        .join(enriched_df.alias("e"), F.col("s.signal_id") == F.col("e.signal_id"), "left_anti")
        .orderBy(F.col("s.score").desc())
        .limit(MAX_SIGNALS)
        .select("s.signal_id", "s.title", "s.body", "s.source", "s.tags")
    )
except Exception:
    # Table doesn't exist yet — all signals are candidates
    candidates = (
        silver_df
        .orderBy(F.col("score").desc())
        .limit(MAX_SIGNALS)
        .select("signal_id", "title", "body", "source", "tags")
    )

candidate_count = candidates.count()
logger.info("Found %d signals to enrich", candidate_count)

if candidate_count == 0:
    logger.info("All signals already enriched — exiting")
    dbutils.notebook.exit("NO_SIGNALS_TO_ENRICH")

# COMMAND ----------

# -- Get API key --
try:
    api_key = get_secret("OPENAI_API_KEY")
except ValueError as exc:
    logger.error("Cannot get OpenAI API key: %s", exc)
    logger.info("LLM enrichment skipped — set OPENAI_API_KEY to enable")
    dbutils.notebook.exit("SKIPPED: No API key configured")

# COMMAND ----------

# -- Process in batches with per-batch checkpointing --
all_rows = candidates.collect()
total_enriched = 0

for batch_start in range(0, len(all_rows), BATCH_SIZE):
    batch = all_rows[batch_start:batch_start + BATCH_SIZE]
    batch_num = (batch_start // BATCH_SIZE) + 1
    total_batches = (len(all_rows) + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info("Processing batch %d/%d (%d signals)", batch_num, total_batches, len(batch))

    results = call_llm_batch(batch, api_key, MODEL)

    if results:
        # Convert results to DataFrame and upsert immediately
        # WHY: Per-batch upsert makes the process resumable. If we crash
        # at batch 5, re-running picks up from where we left off.
        rows = [
            (
                r["signal_id"],
                r["category"],
                r["summary"],
                r["sentiment"],
                r["is_emerging"],
                r["model_used"],
                r["enriched_at"],
            )
            for r in results
        ]
        batch_df = spark.createDataFrame(rows, schema=LLM_ENRICHED_SCHEMA)
        upsert_to_table(
            spark, batch_df,
            TABLE_NAMES["llm_enriched"],
            merge_keys=["signal_id"],
        )
        total_enriched += len(results)
        logger.info("Batch %d complete — %d total enriched so far", batch_num, total_enriched)

logger.info("LLM enrichment complete — %d signals enriched", total_enriched)

# COMMAND ----------

# -- OPTIMIZE --
optimize_table(spark, TABLE_NAMES["llm_enriched"], zorder_cols=["category", "sentiment"])

# COMMAND ----------

# -- Verification --
display(
    spark.table(TABLE_NAMES["llm_enriched"])
    .orderBy(F.col("enriched_at").desc())
    .limit(10)
)

enriched_count = spark.table(TABLE_NAMES["llm_enriched"]).count()
print(f"Total enriched signals: {enriched_count}")

# Category distribution
display(
    spark.table(TABLE_NAMES["llm_enriched"])
    .groupBy("category")
    .agg(F.count("*").alias("count"))
    .orderBy(F.col("count").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Alternative: Pandas UDF
# MAGIC
# MAGIC On a multi-node cluster, the enrichment could be distributed via a Pandas UDF:
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.sql.functions import pandas_udf, PandasUDFType
# MAGIC import pandas as pd
# MAGIC
# MAGIC @pandas_udf(LLM_ENRICHED_SCHEMA, PandasUDFType.GROUPED_MAP)
# MAGIC def enrich_udf(pdf: pd.DataFrame) -> pd.DataFrame:
# MAGIC     """Pandas UDF that calls LLM for each partition.
# MAGIC     Each executor processes its own batch independently.
# MAGIC     """
# MAGIC     results = []
# MAGIC     for _, row in pdf.iterrows():
# MAGIC         result = call_llm_single(row['title'], row['body'], row['source'], row['tags'], api_key, model)
# MAGIC         result['signal_id'] = row['signal_id']
# MAGIC         result['model_used'] = model
# MAGIC         result['enriched_at'] = datetime.utcnow()
# MAGIC         results.append(result)
# MAGIC     return pd.DataFrame(results)
# MAGIC
# MAGIC # Usage:
# MAGIC # enriched = candidates.withColumn("batch_num", (monotonically_increasing_id() % num_batches))
# MAGIC # enriched = enriched.groupBy("batch_num").applyInPandas(enrich_udf, schema=LLM_ENRICHED_SCHEMA)
# MAGIC ```
# MAGIC
# MAGIC **WHY we don't use this on Community Edition:** Single-node means zero parallelism
# MAGIC benefit. The Pandas UDF adds serialization overhead, makes debugging harder, and
# MAGIC prevents per-batch checkpointing. The driver-side loop above is simpler, equally
# MAGIC fast on CE, and more fault-tolerant.

# COMMAND ----------

dbutils.notebook.exit(f"SUCCESS: {total_enriched} signals enriched")
