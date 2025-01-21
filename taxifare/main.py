#!/usr/bin/env python
# coding: utf-8

import os
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data import clean_data
from preprocessor import preprocess_features
from model import initialize_model, compile_model, train_model
from registry import save_model, save_results, load_model

# Global Paths
LOCAL_DATA_PATH = Path(os.path.abspath(os.path.join(os.getcwd(), "..", "data")))
ZIP_FILE_PATH = LOCAL_DATA_PATH / "train.zip"
CSV_FILE_PATH = LOCAL_DATA_PATH / "train.csv"
OUTPUT_CLEANED_PATH = LOCAL_DATA_PATH / "cleaned.csv"
TEST_CSV_PATH = LOCAL_DATA_PATH / "test.csv"
CHUNK_SIZE = 100000

# Load and preprocess data
def load_and_prepare_data():
    """
    Ensure the data is downloaded, extracted, and preprocessed.
    """
    if not CSV_FILE_PATH.exists():
        if ZIP_FILE_PATH.exists():
            print(f"Extracting {ZIP_FILE_PATH}...")
            with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(LOCAL_DATA_PATH)
        else:
            print(f"Downloading data from Google Drive...")
            import gdown
            GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1LjbOSSAdISbvQIia1GbvIKUOxX4ym0v2"
            gdown.download(GOOGLE_DRIVE_URL, str(ZIP_FILE_PATH), quiet=False)
            with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(LOCAL_DATA_PATH)

    if not CSV_FILE_PATH.exists():
        raise FileNotFoundError(f"{CSV_FILE_PATH} is missing. Ensure the download/extraction worked.")

    # Clean and preprocess in chunks
    print("Cleaning and preprocessing data in chunks...")
    with pd.read_csv(CSV_FILE_PATH, chunksize=CHUNK_SIZE) as reader:
        for chunk_id, chunk in enumerate(reader):
            print(f"Processing chunk {chunk_id + 1}...")
            chunk_clean = clean_data(chunk)
            X_chunk = chunk_clean.drop("fare_amount", axis=1)
            y_chunk = chunk_clean[["fare_amount"]]
            X_processed_chunk = preprocess_features(X_chunk)
            chunk_processed = pd.DataFrame(
                np.concatenate((X_processed_chunk, y_chunk), axis=1)
            )
            chunk_processed.to_csv(
                OUTPUT_CLEANED_PATH,
                mode="w" if chunk_id == 0 else "a",
                header=False,
                index=False
            )
    print(f"✅ Data cleaned and saved to {OUTPUT_CLEANED_PATH}")

# Train the model
def train_model_incrementally():
    """
    Train the model incrementally on cleaned data.
    """
    model = None
    metrics_val_list = []

    print("Starting incremental training...")
    with pd.read_csv(OUTPUT_CLEANED_PATH, chunksize=CHUNK_SIZE, header=None) as chunks:
        for chunk_id, chunk in enumerate(chunks):
            if chunk_id >= 250:
                print("Reached maximum number of chunks.")
                break
            print(f"Training on chunk {chunk_id + 1}...")
            split_ratio = 0.1
            train_length = int(len(chunk) * (1 - split_ratio))
            chunk_train = chunk.iloc[:train_length, :].sample(frac=1).to_numpy()
            chunk_val = chunk.iloc[train_length:, :].sample(frac=1).to_numpy()

            X_train = chunk_train[:, :-1]
            y_train = chunk_train[:, -1]
            X_val = chunk_val[:, :-1]
            y_val = chunk_val[:, -1]

            if model is None:
                model = initialize_model(input_shape=X_train.shape[1:])
            model = compile_model(model, learning_rate=0.0005)
            model, history = train_model(
                model, X_train, y_train, batch_size=256, patience=2, validation_data=(X_val, y_val)
            )
            metrics_val_list.append(np.min(history.history['val_mae']))
    save_model(model)
    print(f"✅ Model trained and saved with final MAE: {metrics_val_list[-1]}")

# Evaluate the model
def evaluate_model():
    """
    Evaluate the trained model on test data.
    """
    print("Evaluating the model...")
    df_test = pd.read_csv(TEST_CSV_PATH, parse_dates=["pickup_datetime"])
    df_test.drop(columns=['key'], inplace=True)
    y_test = df_test['fare_amount']
    X_test = df_test.drop(columns=['fare_amount'])
    X_test_processed = preprocess_features(X_test)
    model = load_model()
    y_pred = model.predict(X_test_processed).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ RMSE: {rmse}")

    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Predicted vs Actual Values\nRMSE: {rmse:.2f}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.show()

# Main entry point
if __name__ == "__main__":
    load_and_prepare_data()
    train_model_incrementally()
    evaluate_model()
