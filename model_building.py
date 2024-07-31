import tensorflow as tf
import mlflow
import mlflow.tensorflow
from data_preprocessing import load_and_preprocess_data

def build_and_train_model():
    # Load and preprocess the data
    X_train, X_valid, X_test, y_train, y_valid, y_test, scaler = load_and_preprocess_data()

    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), metrics=['accuracy'])

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Fit the model
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

        # Log parameters
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("epochs", 20)

        # Log metrics
        for epoch, loss in enumerate(history.history['loss']):
            mlflow.log_metric("loss", loss, step=epoch)
        for epoch, val_loss in enumerate(history.history['val_loss']):
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Log the model
        mlflow.tensorflow.log_model(model, "model")

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "MyModel")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    build_and_train_model()