from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import uuid

def get_standard_schema() -> StructType:
    """
    Define the explicit Struct schema for the pipeline.
    Relying on schema inference triggers expensive data passes.
    """
    return StructType([
        StructField("review_id", StringType(), False),
        StructField("user_id", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("raw_text", StringType(), True),
        StructField("timestamp", TimestampType(), True)
    ])

def ingest_data(spark: SparkSession, file_path: str, format: str = "csv") -> DataFrame:
    """
    Ingest data sequentially using lazy evaluation and explicit schema.
    If the source file has different columns, we map them conceptually.
    For demonstration, we assume a CSV with our standard schema natively, 
    OR we cast/rename upon reading if it's the raw Zomato feed.
    """
    schema = get_standard_schema()
    
    # Read the data explicitly bypassing inference
    # Note: If zomato_sandbox.csv has a different schema, we'd read with a lenient schema 
    # and then transform. Here we enforce our standard schema mapped pipeline.
    try:
        df = spark.read.load(
            file_path,
            format=format,
            schema=schema,
            header=True,
            inferSchema=False,
            # We tune CSV reader performance
            timestampFormat="yyyy-MM-dd HH:mm:ss"
        )
    except Exception as e:
        # Fallback reading without schema for the Zomato csv, then shape it
        print("Strict schema mismatch, coercing fields...")
        raw_df = spark.read.csv(file_path, header=True, inferSchema=False)
        
        # Zomato has 'url', 'address', 'name', 'rate', 'reviews_list'
        # We will extract minimal columns to match the target.
        # Generating synthetic IDs where data is missing to ensure non-null constraints
        df = raw_df.withColumn("review_id", F.expr("uuid()")) \
                   .withColumn("user_id", F.col("name")) \
                   .withColumn("product_id", F.col("rest_type")) \
                   .withColumn("raw_text", F.col("reviews_list")) \
                   .withColumn("timestamp", F.current_timestamp()) \
                   .select("review_id", "user_id", "product_id", "raw_text", "timestamp") \
                   .dropna(subset=["raw_text"])

    return df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("IngestionTest").getOrCreate()
    df = ingest_data(spark, "../data/zomato_sandbox.csv")
    df.show(5)
