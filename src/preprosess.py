import os
import sys
import logging
from dotenv import load_dotenv
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

from utils import get_file_from_S3, get_spark

load_dotenv()

hdfs_file_path = os.environ['HDFS_FILE_PATH']

"""
A function to pre-process data.

Args:
bucket_name str Name of S3 bucket
object_key str Name of object file

Returns:
spark_df A pre-processed spark dataframe
"""
def preprocess(bucket_name, object_key):
    print("Preprocessing Pipeline: Started running pipeline for data pre-processing.")

    df = get_file_from_S3(bucket_name, object_key)
    
    spark = get_spark()

    spark_df = spark.createDataFrame(df)

    # Remove duplicates
    spark_df = spark_df.distinct()

    # Create feature column
