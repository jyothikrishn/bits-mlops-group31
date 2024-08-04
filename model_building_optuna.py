import tensorflow as tf
import mlflow
import mlflow.tensorflow
import optuna
from data_preprocessing import load_and_preprocess_data

def build_model(trial, input_shape):
    # Define hyperparameters to tune
    num_units = trial.suggest_int('num_units', 16, 64)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    
    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_units, activation="relu", input_shape=(input_shape,)),
        tf.keras.layers.Dense(1)
    ])

    # Choose optimizer
    if optimizer_name == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['accuracy'])
    
    return model

def objective(trial):
    # Load and preprocess the data
    X_train, X_valid, X_test, y_train, y_valid, y_test, scaler = load_and_preprocess_data()

    model = build_model(trial, X_train.shape[1])

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Fit the model
        history = model.fit(X_train, y_train, epochs=trial.suggest_int('epochs', 10, 50), validation_data=(X_valid, y_valid), verbose=0)

        # Log parameters
        mlflow.log_param("learning_rate", model.optimizer.learning_rate.numpy())
        mlflow.log_param("epochs", len(history.history['loss']))
        mlflow.log_param("num_units", model.layers[0].units)
        mlflow.log_param("optimizer", model.optimizer.__class__.__name__)

        # Log metrics
        val_loss = min(history.history['val_loss'])
        mlflow.log_metric("val_loss", val_loss)

        # Log the model
        mlflow.tensorflow.log_model(model, "model")

        return val_loss

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:8899")
    
    # Create an Optuna study object
    study = optuna.create_study(direction='minimize')
    
    # Optimize the hyperparameters
    study.optimize(objective, n_trials=10)  # Adjust n_trials as needed

    print("Best hyperparameters: ", study.best_params)
    print("Best value: ", study.best_value)
