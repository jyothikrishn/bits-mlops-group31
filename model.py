import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.tensorflow

class MyModel:
    def __init__(self):
        # Load and prepare the California housing dataset
        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full)

        # Scale the features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_valid = self.scaler.transform(X_valid)
        self.X_test = self.scaler.transform(X_test)
        self.y_test = y_test

        # Define the model architecture
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
            tf.keras.layers.Dense(1)
        ])

        # Compile the model
        self.model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3))

        # Fit the model
        self.history = self.model.fit(X_train, y_train, epochs=20,
                                      validation_data=(X_valid, y_valid))

    def predict(self, input_data):
        # Make predictions
        return self.model.predict(input_data)


if __name__ == "__main__":
    # Start an MLflow run
    with mlflow.start_run():
        my_model = MyModel()
        
        # Log parameters
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("epochs", 20)
        
        # Log metrics
        for epoch, loss in enumerate(my_model.history.history['loss']):
            mlflow.log_metric("loss", loss, step=epoch)
        for epoch, val_loss in enumerate(my_model.history.history['val_loss']):
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        # Log the model
        mlflow.tensorflow.log_model(my_model.model, "model")
        
        # Perform inferencing with some sample inputs from the test set
        sample_inputs = my_model.X_test[:5]  # Take the first 5 samples from the test set
        predictions = my_model.predict(sample_inputs)
        
        # Print the predictions
        print("Sample inputs:\n", sample_inputs)
        print("Predictions:\n", predictions)