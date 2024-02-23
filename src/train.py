
import os
import sys
import shutil
import logging
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from dotenv import load_dotenv

from constants import BUCKET_NAME, TEMP_MODEL_PATH
from utils import get_spark, upload

