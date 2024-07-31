import sys
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from data_preprocessing import load_and_preprocess_data

def load_model(run_id):
    # Load the model from MLflow
    model = mlflow.tensorflow.load_model(f"runs:/{run_id}/model")
    return model

def perform_inferencing(run_id):
    # Load and preprocess the data
    _, _, X_test, _, _, y_test, _ = load_and_preprocess_data()

    # Load the model
    model = load_model(run_id)

    # Perform predictions
    predictions = model.predict(X_test[:5])  # Take the first 5 samples from the test set

    # Print the predictions
    print("Sample inputs:\n", X_test[:5])
    print("Predictions:\n", predictions)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    run_id = sys.argv[1]  # Get the run_id from command line arguments
    perform_inferencing(run_id)