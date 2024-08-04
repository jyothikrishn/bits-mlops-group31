import pytest
import tensorflow as tf
from model import MyModel

@pytest.fixture
def model():
    """Fixture to create a new instance of MyModel for each test."""
    return MyModel()

def test_model_initialization(model):
    """Test that the model initializes without errors."""
    assert model is not None, "Failed to initialize the model."

def test_model_predict_shape(model):
    """Test that the model's predict method returns the expected shape."""
    # Adjusting the input shape to match the model's expected input shape (None, 8)
    dummy_input = tf.random.uniform([1, 8])  # Changed from [1, 10] to [1, 8]
    prediction = model.predict(dummy_input)

    assert prediction.shape == (1, 1), "Prediction has unexpected shape."
    
def test_model_predict_values(model):
    """Test that the model's predict method returns values within a specific range."""
    # Adjusting the input shape to match the model's expected input shape (None, 8)
    dummy_input = tf.random.uniform([1, 8])  # Changed from [1, 10] to [1, 8]
    prediction = model.predict(dummy_input)
    # Here, you can add assertions to check if the prediction values are within the expected range
    