import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

class MyModel:
    def __init__(self):
        # Load and prepare the California housing dataset
        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

        # Define the model architecture
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
            tf.keras.layers.Dense(1)
        ])

        # Compile the model
        self.model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.SGD(lr=1e-3))

        # Fit the model
        self.history = self.model.fit(X_train, y_train, epochs=20,
                                      validation_data=(X_valid, y_valid))

    def predict(self, input_data):
        # Make predictions
        return self.model.predict(input_data)

# Example usage
if __name__ == "__main__":
    my_model = MyModel()
    # Assuming `input_data` is a numpy array of shape (None, 8) as per the California housing dataset
    # input_data = ...
    # print(my_model.predict(input_data))