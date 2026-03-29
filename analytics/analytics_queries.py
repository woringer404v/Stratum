# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 08 — Analytics Queries
# MAGIC
# MAGIC **What this notebook does:** Showcase 10 SQL analytics queries against the Gold layer,
# MAGIC plus Delta Lake time travel and history demonstrations. These are the queries a
# MAGIC Databricks SE would actually demo to a customer.
# MAGIC
# MAGIC **Inputs:** All Gold layer tables + Silver `tech_signals` + `data_quality_log`
# MAGIC
# MAGIC **Outputs:** Display results only (no table writes)
# MAGIC
# MAGIC **Dependencies:** All upstream notebooks must have run at least once

# COMMAND ----------

import logging
from config import TABLE_NAMES, apply_spark_config, setup_logging

logger = setup_logging()

# COMMAND ----------

# -- Widget definitions --
dbutils.widgets.text("date_from", "2026-01-01", "Start date filter")
dbutils.widgets.text("date_to", "2026-12-31", "End date filter")
dbutils.widgets.dropdown("source_filter", "all", ["all", "hackernews", "github", "arxiv", "stackoverflow"], "Source filter")

DATE_FROM = dbutils.widgets.get("date_from")
DATE_TO = dbutils.widgets.get("date_to")
SOURCE_FILTER = dbutils.widgets.get("source_filter")

# COMMAND ----------

apply_spark_config(spark)

# Build source filter clause for SQL queries
source_clause = "" if SOURCE_FILTER == "all" else f"AND source = '{SOURCE_FILTER}'"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 1 — Top 20 Trending Signals This Week
# MAGIC The highest-impact signals across all sources, ranked by a composite score that
# MAGIC weighs engagement, tag richness, and recency.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   rank_in_source AS rank,
# MAGIC   source,
# MAGIC   title,
# MAGIC   score AS raw_score,
# MAGIC   ROUND(composite_score, 2) AS composite_score,
# MAGIC   url
# MAGIC FROM stratum_gold.trending_signals
# MAGIC WHERE date >= date_sub(current_date(), 7)
# MAGIC ORDER BY composite_score DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 2 — Fastest Accelerating Technologies
# MAGIC Terms with the steepest week-over-week growth. These are the technologies gaining
# MAGIC momentum right now — the ones a Databricks SE would want to know about before
# MAGIC walking into a customer meeting.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   term,
# MAGIC   this_week_count,
# MAGIC   last_week_count,
# MAGIC   ROUND(velocity, 2) AS velocity,
# MAGIC   CASE WHEN is_accelerating THEN 'ACCELERATING' ELSE 'stable' END AS status
# MAGIC FROM stratum_gold.tech_velocity
# MAGIC WHERE is_accelerating = true
# MAGIC   AND this_week_count >= 3
# MAGIC ORDER BY velocity DESC
# MAGIC LIMIT 15

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 3 — Source Activity Comparison
# MAGIC How do the four data sources compare in volume and average signal quality over the
# MAGIC past 30 days? This shows whether our pipeline is healthy and balanced.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   source,
# MAGIC   SUM(signal_count) AS total_signals,
# MAGIC   ROUND(AVG(avg_score), 1) AS avg_score,
# MAGIC   MAX(max_score) AS peak_score,
# MAGIC   SUM(unique_authors) AS total_authors
# MAGIC FROM stratum_gold.source_summary
# MAGIC WHERE date >= date_sub(current_date(), 30)
# MAGIC GROUP BY source
# MAGIC ORDER BY total_signals DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 4 — Daily Signal Volume Trend
# MAGIC Time series of signal volume per source — useful for spotting spikes (a big tech
# MAGIC announcement) or drops (API issues, rate limit hits).

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   date,
# MAGIC   source,
# MAGIC   signal_count
# MAGIC FROM stratum_gold.source_summary
# MAGIC WHERE date >= date_sub(current_date(), 30)
# MAGIC ORDER BY date, source

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 5 — Top 10 Terms Per Source
# MAGIC What is each community talking about the most? Reveals whether HN, GitHub, arXiv,
# MAGIC and SO are focused on the same technologies or diverging.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   source,
# MAGIC   term,
# MAGIC   frequency,
# MAGIC   rank_in_source
# MAGIC FROM stratum_gold.tech_term_frequency
# MAGIC WHERE rank_in_source <= 10
# MAGIC   AND date = (SELECT MAX(date) FROM stratum_gold.tech_term_frequency)
# MAGIC ORDER BY source, rank_in_source

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 6 — Cross-Source Term Overlap
# MAGIC Technologies that appear in 3+ sources simultaneously are signals of genuine
# MAGIC industry-wide interest, not just community-specific hype.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   term,
# MAGIC   COUNT(DISTINCT source) AS source_count,
# MAGIC   SUM(frequency) AS total_mentions,
# MAGIC   COLLECT_SET(source) AS sources
# MAGIC FROM stratum_gold.tech_term_frequency
# MAGIC WHERE date >= date_sub(current_date(), 7)
# MAGIC GROUP BY term
# MAGIC HAVING COUNT(DISTINCT source) >= 3
# MAGIC ORDER BY total_mentions DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 7 — Emerging Technology Radar
# MAGIC Technologies flagged by the LLM as emerging this week, grouped by category.
# MAGIC This is the "early warning system" — what should we pay attention to next?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   e.category,
# MAGIC   COUNT(*) AS emerging_count,
# MAGIC   COLLECT_LIST(s.title) AS sample_titles
# MAGIC FROM stratum_gold.llm_enriched e
# MAGIC JOIN stratum_silver.tech_signals s ON e.signal_id = s.signal_id
# MAGIC WHERE e.is_emerging = true
# MAGIC   AND e.enriched_at >= date_sub(current_date(), 7)
# MAGIC GROUP BY e.category
# MAGIC ORDER BY emerging_count DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 8 — Sentiment Distribution by Source
# MAGIC Are certain communities more optimistic or frustrated? This reveals the emotional
# MAGIC landscape of each data source — useful for understanding community health.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   s.source,
# MAGIC   e.sentiment,
# MAGIC   COUNT(*) AS signal_count,
# MAGIC   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY s.source), 1) AS pct
# MAGIC FROM stratum_gold.llm_enriched e
# MAGIC JOIN stratum_silver.tech_signals s ON e.signal_id = s.signal_id
# MAGIC WHERE e.category != 'error'
# MAGIC GROUP BY s.source, e.sentiment
# MAGIC ORDER BY s.source, signal_count DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 9 — Author Leaderboard
# MAGIC Who are the most prolific, high-quality contributors across all sources?
# MAGIC Authors appearing multiple times with high average scores are domain experts.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   author,
# MAGIC   COUNT(*) AS signal_count,
# MAGIC   ROUND(AVG(score), 1) AS avg_score,
# MAGIC   COLLECT_SET(source) AS active_sources,
# MAGIC   MAX(created_at) AS latest_signal
# MAGIC FROM stratum_silver.tech_signals
# MAGIC WHERE author IS NOT NULL AND author != 'anonymous' AND author != 'unknown'
# MAGIC GROUP BY author
# MAGIC HAVING COUNT(*) >= 3
# MAGIC ORDER BY avg_score DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query 10 — Data Quality Dashboard
# MAGIC Operational health monitoring: null rates, duplicate rates, and freshness per source
# MAGIC over the last 7 days. This is the query that tells you if the pipeline is broken.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   source,
# MAGIC   DATE(checked_at) AS check_date,
# MAGIC   SUM(total_rows) AS total_rows,
# MAGIC   SUM(duplicate_count) AS total_duplicates,
# MAGIC   ROUND(AVG(freshness_hours), 1) AS avg_freshness_hours,
# MAGIC   SUM(CASE WHEN quality_passed THEN 1 ELSE 0 END) AS checks_passed,
# MAGIC   COUNT(*) AS total_checks
# MAGIC FROM stratum_silver.data_quality_log
# MAGIC WHERE checked_at >= date_sub(current_date(), 7)
# MAGIC GROUP BY source, DATE(checked_at)
# MAGIC ORDER BY check_date DESC, source

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Delta Lake Time Travel Demonstrations
# MAGIC
# MAGIC Delta Lake stores full version history. Time travel lets us query any previous
# MAGIC state of a table — essential for audit trails, debugging, and reproducibility.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time Travel Query 1 — VERSION AS OF
# MAGIC Compare the current trending table with a previous version to see how rankings shifted.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Current version
# MAGIC DESCRIBE HISTORY stratum_gold.trending_signals

# COMMAND ----------

spark.sql("""
    SELECT version, timestamp, operation, operationMetrics
    FROM (DESCRIBE HISTORY stratum_gold.trending_signals)
    ORDER BY version DESC
    LIMIT 5
""").display()

# COMMAND ----------

# WHY: Time travel by version number. We query the earliest available version
# to show how the data looked at the beginning vs now. In a demo, this would
# show "here's what was trending 3 ingestion runs ago".
earliest_version = spark.sql(
    "SELECT MIN(version) FROM (DESCRIBE HISTORY stratum_gold.trending_signals)"
).first()[0]

if earliest_version is not None:
    spark.sql(f"""
        SELECT source, title, ROUND(composite_score, 2) AS composite_score
        FROM stratum_gold.trending_signals VERSION AS OF {earliest_version}
        ORDER BY composite_score DESC
        LIMIT 10
    """).display()
else:
    print("No version history available yet — run the pipeline at least twice")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time Travel Query 2 — TIMESTAMP AS OF
# MAGIC Query the state of the source summary table as of a specific time.

# COMMAND ----------

# Query the table as it was 1 hour ago (or earliest available)
spark.sql("""
    SELECT *
    FROM stratum_gold.source_summary
    TIMESTAMP AS OF date_sub(current_timestamp(), 1)
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delta History — Full Audit Trail
# MAGIC Every write operation is recorded. This is what makes Delta Lake compliance-ready.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show the complete audit trail for the Silver signals table
# MAGIC DESCRIBE HISTORY stratum_silver.tech_signals

# COMMAND ----------

dbutils.notebook.exit("SUCCESS: All analytics queries executed")
