# Scalable Credit Card Fraud Detection System Using Apache Spark

## Overview

This project involves the development of a scalable credit card fraud detection system utilizing Apache Spark for distributed data processing. The system processes large datasets to detect fraudulent transactions, leveraging machine learning techniques to achieve high accuracy, recall, and F1-score.

## Features

- **Data Preprocessing:** The system includes a robust data preprocessing pipeline that standardizes features, removes duplicates, and handles large-scale data efficiently using PySpark.
- **Model Training:** A RandomForestClassifier is trained on the processed data to detect fraudulent transactions. The model is evaluated using metrics such as accuracy, recall, and F1-score.
- **Inference Pipeline:** The model is applied to new data batches for real-time fraud detection, with results securely stored and ready for further analysis.

## Performance Metrics

- **Accuracy:** 92%
- **Recall:** 88%
- **F1-Score:** 90%

## Technologies Used

