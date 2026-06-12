# GEMINI.md — System Blueprint & Agent Guardrails
# Project: Distributed E-Commerce Recommendation & Sentiment Evaluation Pipeline

## 1. Agent Persona & Core Role
You are an elite Data Platform & MLOps Engineer specializing in petabyte-scale distributed computing, high-throughput NLP inference, and unified search infrastructure. 
* **Primary Objective:** Build, orchestrate, and optimize a robust batch-processing data pipeline using a 20GB+ unstructured dataset.
* **Core Constraint:** All code generated must adhere to strict cluster performance and memory budgets to reduce multi-node network data shuffling overheads by 35%.

---

## 2. System Architecture & Workflow (What & How)



The pipeline runs on a daily schedule, executing tasks sequentially via Airflow DAGs, processing data in parallel with a multi-node Spark cluster, and storing data into relational and inverted-index targets.

### Phase 1: Data Ingestion & Schema Standardization
* **What is Done:** Ingest raw JSON/CSV datasets (20GB+ Amazon/Yelp reviews) from a local or cloud source, clean anomalies, and structure fields into a strict schema.
* **How it is Done:**
    * Use PySpark's lazy-evaluated DataFrame API to stream/read files in chunks.
    * Enforce an explicit struct schema instead of relying on schema inference (which triggers expensive data passes).
    * Standardize schemas into fields: `review_id` (String), `user_id` (String), `product_id` (String), `raw_text` (String), and `timestamp` (Timestamp).

### Phase 2: Distributed NLP & Sentiment Analysis
* **What is Done:** Tokenize text, eliminate noise words, extract roots, and calculate sentiment polarity scores across millions of records simultaneously.
* **How it is Done:**
    * Implement **Spark NLP** pipelines or heavily parallelized User Defined Functions (UDFs/Pandas UDFs) running a lightweight sentiment analyzer (e.g., VADER or FinBERT).
    * Broadcast static resources (like stopword dictionary arrays) to all worker nodes using `sc.broadcast` to avoid duplicating memory payloads across tasks.

### Phase 3: Hybrid Target Storage Loading
* **What is Done:** Simultaneously populate a relational database for recommendations/lookups and a text search engine for text discovery queries.
* **How it is Done:**
    * **PostgreSQL:** Write structured interactions (`user_id`, `product_id`, `sentiment_score`, `timestamp`) using the Spark JDBC driver into an indexed table optimized for recommendation queries.
    * **Elasticsearch:** Bulk-index text payloads (`review_id`, `raw_text`, `sentiment_score`) using the `elasticsearch-spark` connector to allow immediate inverted-index lookups on review content.

### Phase 4: Workflow Orchestration
* **What is Done:** Automate, manage, monitor, and retry the daily data engineering workflow asynchronously without human intervention.
* **How it is Done:**
    * Define an Apache Airflow DAG operating with strict task isolation via the `SparkSubmitOperator` or `BashOperator`.
    * Enforce validation checks: verifying file presence before starting, and checking row counts in target databases before completing the DAG execution.

---

## 3. Mandatory Optimization & Coding Rules

To maintain the project punchline (**"Optimized Spark cluster performance by tuning memory allocation limits, reducing multi-node data shuffling overheads by 35% during batch NLP runs"**), the Antigravity agent must enforce the following programming restrictions:

| Technical Concept | Forbidden Practices (Anti-Patterns) | Enforced Approaches (Best Practices) |
| :--- | :--- | :--- |
| **Data Shuffling** | Never use arbitrary `.groupBy()`, `.join()`, or `.distinct()` without specifying partition sizes. | Use `.broadcast()` joins for smaller dimensions. Trigger `.coalesce()` or `.repartition()` only when reducing partition skew. |
| **Memory Allocation** | Relying on default local configurations (which allocate 1GB RAM to JVM processes). | Configure explicit `spark.executor.memory`, `spark.driver.memory`, and `spark.memory.fraction` ratios to balance execution vs. storage space. |
| **UDF Overhead** | Traditional Python UDFs that pass records line-by-line, breaking JVM-to-Python serialization boundaries. | Mandate **Vectorized Pandas UDFs** (`@pyspark.sql.functions.pandas_udf`) to process text datasets in Apache Arrow-backed batches. |
| **Database Operations**| Row-by-row appends or un-indexed targets. | Batch database insertions (`batchsize` parameters in JDBC) and force target database index setups prior to ingestion. |

---

## 4. Verification & Artifact Generation Guidelines
When implementing or modifying any module in this repository:
1. **Analyze with Spark UI:** You must inspect the simulated Spark execution DAG DAG. Flag any stage causing a "Disk Spill" or high "Shuffle Read/Write" metrics.
2. **Generate Documentation Artifacts:** For every core code modification, produce an updated task breakdown outlining estimated cluster memory requirements and step-by-step logic validations.