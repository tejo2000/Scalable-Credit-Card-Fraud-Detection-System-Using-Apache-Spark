
import os
import sys
import shutil
import logging
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from dotenv import load_dotenv

from constants import BUCKET_NAME, TEMP_MODEL_PATH
from utils import get_spark, upload

load_dotenv()

def train_model(df):
    train, test = df.randomSplit([0.7, 0.3], seed = 2018)

    rf_clf = RandomForestClassifier(featuresCol = 'scaledFeatures', labelCol = 'Class')
    rfModel = rf_clf.fit(train)

    predictions = rfModel.transform(test)

    evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction")

    accuracy = evaluator.evaluate(predictions)

    print("Training Piepline: Training of Model completed. Overal accuracy is {0}".format(accuracy))

