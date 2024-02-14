import os
import sys
import shutil
import logging
import uuid
import pandas as pd
from dotenv import load_dotenv
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import RandomForestClassificationModel

from constants import BUCKET_NAME, TEMP_MODEL_PATH, PREDICTION_TABLE
