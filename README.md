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

- **Apache Spark & PySpark:** For distributed data processing and machine learning.
- **HDFS (Hadoop Distributed File System):** For storing and accessing large datasets.
- **RandomForestClassifier:** For building the fraud detection model.
- **Pandas:** For handling and manipulating data during post-processing.
- **Python:** The primary programming language used throughout the project.
- **Dotenv:** For managing environment variables.

## Project Structure

- **`constants.py`**: Contains constant values such as bucket names, model paths, and table names.
- **`inference.py`**: Handles the inference pipeline, including data preprocessing, model loading, and prediction storage.
- **`preprocess.py`**: Responsible for the data preprocessing pipeline, including feature engineering and scaling.
- **`train.py`**: Manages the training process of the machine learning model and evaluates its performance.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
