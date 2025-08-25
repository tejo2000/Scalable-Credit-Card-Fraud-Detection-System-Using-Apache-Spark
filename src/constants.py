BUCKET_NAME = "fraud-detection-project"
TEMP_MODEL_PATH = "model"
PREDICTION_TABLE="fraud_prediction"
# refactor: extract Spark session creation into get_spark() to avoid re-init
