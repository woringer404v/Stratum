# Stratum

**Multi-source tech intelligence pipeline built natively on Databricks.**

Stratum ingests data from four public APIs (Hacker News, GitHub, arXiv, Stack Overflow), processes it through a Medallion architecture using Delta Lake, enriches it with LLM-powered analysis, and serves insights via a SQL analytics layer — all running on Databricks Community Edition.

---

## Why I Built This

The tech industry generates signals across dozens of platforms, but no single source tells the whole story. Stratum unifies these fragmented signals into a single, queryable intelligence layer so you can answer questions like "what technologies are accelerating across research, open source, and practitioner communities simultaneously?" It's built as a production-minded data engineering project that demonstrates Databricks-native patterns end to end.

---

## Architecture

```
                            ┌─────────────────────────────────────────────────┐
                            │              STRATUM  PIPELINE                  │
                            └─────────────────────────────────────────────────┘

  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Hacker News │   │    GitHub    │   │    arXiv     │   │Stack Overflow│
  │  Firebase API│   │  Search API  │   │   Atom API   │   │    SE API    │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                  │                  │
         ▼                  ▼                  ▼                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        B R O N Z E   L A Y E R                         │
  │  stratum_bronze.hackernews_raw  │  github_raw  │  arxiv_raw  │  so_raw │
  │                                                                        │
  │  • Append-only Delta tables     • Ingestion metadata (_ingested_at,    │
  │  • _raw_json for schema-on-read   _source, _batch_id)                  │
  │  • Checkpoint table for simulated Auto Loader                          │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                       S I L V E R   L A Y E R                          │
  │  stratum_silver.tech_signals (unified schema)                          │
  │                                                                        │
  │  signal_id │ source │ title │ body │ url │ author │ score │ created_at │
  │  tags (array) │ language │ _silver_processed_at                        │
  │                                                                        │
  │  • MERGE upsert (idempotent)   • Type casting & null handling          │
  │  • Deduplication by signal_id  • HTML stripping (SO bodies)            │
  │  • Quality metrics → data_quality_log                                  │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 │
                        ┌────────┴────────┐
                        ▼                 ▼
  ┌──────────────────────────┐  ┌──────────────────────────┐
  │    G O L D   L A Y E R   │  │   L L M   E N R I C H    │
  │                          │  │                          │
  │  • tech_term_frequency   │  │  • OpenAI ChatCompletion │
  │  • trending_signals      │  │  • Category tagging      │
  │  • source_summary        │  │  • Sentiment analysis    │
  │  • tech_velocity         │  │  • Emerging-tech flag    │
  │                          │  │  • 1-sentence summary    │
  │  Window functions, aggs  │  │  → gold.llm_enriched     │
  │  OPTIMIZE + ZORDER       │  │                          │
  └──────────┬───────────────┘  └──────────┬───────────────┘
             │                             │
             └──────────┬──────────────────┘
                        ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    A N A L Y T I C S   L A Y E R                       │
  │                                                                        │
  │  10 SQL queries  •  Delta time travel  •  DESCRIBE HISTORY             │
  │  Cross-source analysis  •  Velocity tracking  •  Quality dashboard     │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Databricks Features Used

| Feature | Where Used | Why |
|---------|-----------|-----|
| **Delta Lake** | Every table in every layer | ACID transactions, time travel, schema enforcement |
| **Medallion Architecture** | Bronze → Silver → Gold layers | Clean separation of raw, cleaned, and analytics-ready data |
| **MERGE (upsert)** | Silver writes, Gold writes | Idempotent re-runs — no duplicates on retry |
| **Delta Time Travel** | Analytics notebook (VERSION AS OF, TIMESTAMP AS OF) | Audit trail and historical comparison |
| **DESCRIBE HISTORY** | Analytics notebook | Full operation audit log per table |
| **OPTIMIZE + ZORDER** | All Gold tables | Query performance via file compaction and co-location |
| **Auto Optimize** | All tables (`optimizeWrite`, `autoCompact`) | Prevents small-file problem from frequent appends |
| **Schema Evolution** | Bronze writes (`mergeSchema`) | New API fields land without breaking the pipeline |
| **dbutils.widgets** | Every notebook | Parameterised notebooks for flexible execution |
| **dbutils.notebook.run** | Orchestration notebook | Sequential notebook execution with error handling |
| **dbutils.secrets** | Config module | Secure secret management (with env var fallback) |
| **Hive Metastore** | All table references (`database.table`) | Table discovery without Unity Catalog (CE constraint) |
| **Spark SQL** | Analytics queries (`%sql` magic) | Native SQL analytics on Delta tables |
| **PySpark Window Functions** | Gold table computations | Ranking, velocity, composite scoring |

---

## Design Decisions

### 1. Append-only Bronze, MERGE Silver/Gold
Bronze is an immutable audit trail of exactly what the APIs returned. If we discover a transformation bug or an API changes its response format, we can reprocess from Bronze without re-fetching. Silver and Gold use MERGE for idempotent upserts — re-running a batch never creates duplicates.

### 2. `signal_id = md5(concat(source, native_id))`
A deterministic surrogate key that is stable across runs and unique across sources. Using `concat(source, native_id)` prevents cross-source collisions (e.g., HN item id=1 vs GitHub repo id=1). MD5 is fast and collision-resistant at our scale.

### 3. Simulated Auto Loader via Checkpoint Table
Community Edition lacks structured streaming and Auto Loader. We simulate the three key guarantees — incremental processing, exactly-once semantics, and checkpoint recovery — using a `_checkpoints` Delta table. Each source stores a high-water mark (max ID or max timestamp). The checkpoint table itself uses MERGE for atomic updates.

### 4. Driver-side LLM Calls, Not a Pandas UDF
Community Edition is single-node — a Pandas UDF adds serialization overhead with zero parallelism benefit. Driver-side iteration is simpler, easier to debug, and allows per-batch checkpointing (if the notebook crashes at batch 5, re-running resumes from batch 6 via LEFT ANTI JOIN). The UDF pattern is documented in the notebook for production reference.

### 5. GitHub Search API with `created:>DATE sort:stars`
GitHub has no official trending endpoint. We use the search API to find recently-created repositories sorted by stars — this surfaces repos gaining traction rather than established projects with static star counts.

### 6. Quality Gate Warns But Doesn't Fail
Data quality issues (high null rates, staleness) are logged to `data_quality_log` but don't halt the pipeline. Hard failures on data quality would require manual intervention with no self-healing benefit — the data is already in Bronze and can be reprocessed.

### 7. `spark.sql.shuffle.partitions = 8`
Community Edition has limited cores and memory. The default 200 partitions creates excessive small files and shuffle overhead on a single node. 8 partitions is appropriate for our data volumes (thousands of rows, not millions).

---

## What I'd Do Differently at Scale

### 1. Replace Simulated Checkpoints with Auto Loader
On a production Databricks workspace, I'd use Auto Loader (`cloudFiles` format) with schema inference enabled. This gives exactly-once guarantees natively, handles schema evolution automatically, and scales to millions of files without checkpoint management code.

### 2. Replace Sequential Orchestration with Databricks Workflows
The orchestration notebook uses `dbutils.notebook.run()` sequentially. In production, I'd use Databricks Workflows (Jobs API) with a DAG of tasks — ingestion notebooks run in parallel, Silver depends on all four, Gold depends on Silver. This cuts pipeline runtime by 3-4x and adds built-in alerting, retry policies, and SLA monitoring.

### 3. Replace Hive Metastore with Unity Catalog
Unity Catalog provides table-level and column-level access control, data lineage tracking, cross-workspace governance, and audit logging. On Community Edition we use the Hive metastore because Unity Catalog isn't available, but every `database.table` reference would migrate directly.

---

## Project Structure

```
Stratum/
├── README.md                              # This file
├── config.py                              # Central configuration
├── utils/
│   ├── api_utils.py                       # HTTP client with retry, pagination
│   ├── delta_utils.py                     # Delta Lake operations, checkpoints
│   └── quality_utils.py                   # Data quality checks and logging
├── ingestion/
│   ├── ingest_hackernews.py               # Databricks notebook
│   ├── ingest_github.py                   # Databricks notebook
│   ├── ingest_arxiv.py                    # Databricks notebook
│   └── ingest_stackoverflow.py            # Databricks notebook
├── silver/
│   └── bronze_to_silver.py                # Unified Silver transform
├── gold/
│   ├── silver_to_gold.py                  # Gold table computations
│   └── llm_enrichment.py                  # LLM-powered enrichment
├── analytics/
│   └── analytics_queries.py               # 10 SQL showcase queries
└── orchestration/
    └── run_pipeline.py                    # Master pipeline orchestrator
```

---

## Setup Instructions

### Prerequisites
- A Databricks Community Edition account ([community.cloud.databricks.com](https://community.cloud.databricks.com))
- An OpenAI API key (optional — LLM enrichment is skipped if not configured)

### Step-by-Step

1. **Create a Databricks cluster** in Community Edition (Runtime 14.x or 15.x, default settings)

2. **Import the repository** into your Databricks workspace:
   - Workspace → Users → your_user → Import
   - Upload the `Stratum/` folder or connect to the Git repo

3. **Configure secrets** (optional, for LLM enrichment):
   ```python
   # In a notebook cell:
   # Option A: Databricks secrets (recommended)
   # Create secret scope 'stratum' and add key 'OPENAI_API_KEY'

   # Option B: Environment variable
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."
   ```

4. **Run the pipeline**:
   - Open `orchestration/run_pipeline.py`
   - Click "Run All"
   - Default settings will run all sources in incremental mode
   - First run takes ~10-15 minutes (API fetching is the bottleneck)

5. **Explore results**:
   - Open `analytics/analytics_queries.py` for SQL analytics
   - Check table row counts in the orchestration summary
   - Query any Gold table directly via SQL

### Widget Configuration

| Widget | Default | Description |
|--------|---------|-------------|
| `run_mode` | incremental | `incremental` or `full` — full reprocesses all data |
| `skip_llm` | false | Set to `true` if no OpenAI key configured |
| `batch_size` | 200 | Number of HN stories per run |
| `max_signals` | 50 | Max signals to LLM-enrich per run |

---

## Sample Output

### Gold: trending_signals
```
┌──────────┬──────────────────────────────────────┬───────┬─────────────────┬──────┐
│ source   │ title                                │ score │ composite_score │ rank │
├──────────┼──────────────────────────────────────┼───────┼─────────────────┼──────┤
│ hackernews│ Show HN: Open-source LLM framework  │  342  │     47.82       │  1   │
│ github   │ lightning-fast-vector-db             │  1205 │     38.91       │  2   │
│ arxiv    │ Efficient Fine-Tuning of LLMs...     │    0  │      0.69       │  1   │
│ stackoverflow│ How to deploy Spark on K8s?      │   15  │      3.21       │  3   │
└──────────┴──────────────────────────────────────┴───────┴─────────────────┴──────┘
```

### Gold: llm_enriched
```
┌────────────┬──────────────────────────────────────────┬───────────┬───────────┐
│ category   │ summary                                  │ sentiment │ emerging  │
├────────────┼──────────────────────────────────────────┼───────────┼───────────┤
│ AI/ML      │ New framework for efficient LLM serving  │ positive  │ true      │
│ Data Eng.  │ Discussion on Spark vs Flink for stream  │ neutral   │ false     │
│ DevOps     │ Kubernetes operator for ML pipelines     │ positive  │ true      │
└────────────┴──────────────────────────────────────────┴───────────┴───────────┘
```

---

## License

This project is for portfolio and educational purposes.
