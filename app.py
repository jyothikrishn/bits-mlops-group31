import mlflow
import mlflow.tensorflow
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model from the MLflow Model Registry
model_name = "MyModel"
model_version = 1  # Specify the version of the model you want to use
model = mlflow.tensorflow.load_model(f"models:/{model_name}/{model_version}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predictions = model.predict(data['input'])
    return jsonify(predictions.tolist())

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    app.run(host='0.0.0.0', port=5001)