from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'data_platform_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ecommerce_sentiment_pipeline',
    default_args=default_args,
    description='Daily Batch Spark Pipeline for Sentiment NLP and Recommendation DB',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['spark', 'nlp', 'batch'],
)

# Phase 4 Task 1: Validation Check - Raw File Presence
# Verifying dataset presence before starting the expensive Spark cluster
raw_data_path = "/Users/ayushrawat/SentiCart/data/zomato_sandbox.csv"
file_presence_check = BashOperator(
    task_id='verify_file_presence',
    bash_command=f'test -f {raw_data_path} || (echo "Raw file missing!" && exit 1)',
    dag=dag,
)

# Phase 4 Task 2: Isolated Spark Execution via SparkSubmitOperator
# We submit our nlp_processing.py. We also specify packages here if needed,
# though we already put it in the builder inside the script. We pass it here 
# just to be safe as well for Spark Submit context.
spark_submit_task = SparkSubmitOperator(
    task_id='run_spark_sentiment_job',
    application='/Users/ayushrawat/SentiCart/src/nlp_processing.py',
    name='ECommerceSentimentPipeline',
    packages='org.elasticsearch:elasticsearch-spark-30_2.12:8.8.1,org.postgresql:postgresql:42.6.0',
    executor_memory='4g',
    driver_memory='2g',
    conf={
        'spark.memory.fraction': '0.6',
        'spark.sql.shuffle.partitions': '50',
    },
    conn_id='spark_default',
    verbose=True,
    dag=dag,
)

# Phase 4 Task 3: Validation Check - DB Row Counts
# We check row counts using Docker exec to postgres to see if it wrote
db_count_check = BashOperator(
    task_id='verify_db_row_counts',
    # Since postgres is running in docker-compose, we use docker exec.
    bash_command='docker exec postgres_db psql -U admin -d ecommerce_analytics -c "SELECT COUNT(*) FROM recommendation_interactions;"',
    dag=dag,
)

# Set dependencies
file_presence_check >> spark_submit_task >> db_count_check
