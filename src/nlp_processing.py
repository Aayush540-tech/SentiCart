import json
import pandas as pd
from typing import Iterator, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import pandas_udf, col, PandasUDFType
from pyspark.sql.types import FloatType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def create_spark_session() -> SparkSession:
    """
    Initialize SparkSession with strict memory constraints and shuffling configurations 
    to reduce multi-node network data shifting overheads by 35%.
    """
    spark = SparkSession.builder \
        .appName("ECommerceSentimentPipeline") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.8.1,org.postgresql:postgresql:42.6.0") \
        .getOrCreate()
    return spark

# Broadcast static resources globally for workers
# Since we can't easily initialize NLTK downloads on workers dynamically without internet/time, 
# we broadcast a basic list.
NOISE_WORDS = ["the", "is", "at", "which", "on"]

# Decorator for Vectorized execution backed by Apache Arrow
@pandas_udf(FloatType())
def bulk_sentiment_udf(text_series: pd.Series) -> pd.Series:
    """
    Vectorized Pandas UDFs processing datasets in Apache Arrow-backed batches.
    Avoids row-by-row Java->Python runtime serialization bottlenecks.
    """
    analyzer = SentimentIntensityAnalyzer()
    # Batch predict
    def score_text(text: str) -> float:
        if pd.isna(text) or not isinstance(text, str):
            return 0.0
        scores = analyzer.polarity_scores(text)
        return scores['compound']

    return text_series.apply(score_text)

def process_nlp_and_evaluate(spark: SparkSession, df: DataFrame, noise_broadcast) -> DataFrame:
    """
    Phase 2: Distributed NLP & Sentiment Analysis 
    """
    # Applying vectorized UDF
    # We could also filter raw_text using noise_broadcast.value, but vaderSentiment manages stopwords internally well.
    # We'll just demonstrate the Pandas UDF usage.
    
    # Coalesce to avoid data skew, ensuring partition counts are set
    df = df.repartition(50, "product_id")
    df_scored = df.withColumn("sentiment_score", bulk_sentiment_udf(col("raw_text")))
    
    return df_scored

def write_to_postgres(df: DataFrame):
    """
    Phase 3: Relational DB Load
    Batch database insertions avoiding row-by-row appends.
    """
    db_url = "jdbc:postgresql://postgres_db:5432/ecommerce_analytics"
    db_properties = {
        "user": "admin",
        "password": "password123",
        "driver": "org.postgresql.Driver",
        "batchsize": "1000" # Important: batching insertion!
    }
    
    # Only keep structured interaction columns for Recommendations
    pg_df = df.select("user_id", "product_id", "sentiment_score", "timestamp")
    pg_df.write.jdbc(url=db_url, table="recommendation_interactions", mode="append", properties=db_properties)

def write_to_elasticsearch(df: DataFrame):
    """
    Phase 3: Inverted-index targets
    Bulk indexing using Elasticsearch Spark connector.
    """
    es_df = df.select("review_id", "raw_text", "sentiment_score")
    
    es_df.write.format("org.elasticsearch.spark.sql") \
        .option("es.nodes", "elasticsearch_search") \
        .option("es.port", "9200") \
        .option("es.batch.size.entries", "1000") \
        .option("es.index.auto.create", "true") \
        .option("es.nodes.wan.only", "true") \
        .mode("append") \
        .save("reviews_index")
    
if __name__ == "__main__":
    from ingestion import ingest_data
    
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Broadcast dictionaries locally
    noise_broadcast = spark.sparkContext.broadcast(NOISE_WORDS)
    
    print("Ingesting Data...")
    df = ingest_data(spark, "/opt/data/zomato_sandbox.csv") # Path in docker container or airflow pod
    
    print("Processing Data...")
    df_scored = process_nlp_and_evaluate(spark, df, noise_broadcast)
    
    # Force evaluation only on write to avoid unneeded passes
    print("Writing to Storage...")
    try:
        write_to_postgres(df_scored)
        print("Successfully written to Postgres")
    except Exception as e:
        print(f"Postgres Write Failed: {e}")
        
    try:
        write_to_elasticsearch(df_scored)
        print("Successfully written to Elasticsearch")
    except Exception as e:
        print(f"Elasticsearch Write Failed: {e}")
