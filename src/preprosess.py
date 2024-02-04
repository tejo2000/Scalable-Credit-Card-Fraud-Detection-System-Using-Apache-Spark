import os
import sys
import logging
from dotenv import load_dotenv
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

from utils import get_file_from_S3, get_spark
