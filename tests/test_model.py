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
    # Assuming the model expects input shape (None, 10) and outputs shape (None, 1)
    dummy_input = tf.random.uniform([1, 10])
    prediction = model.predict(dummy_input)
    assert prediction.shape == (1, 1), "Prediction shape is incorrect."

def test_model_predict_values(model):
    """Test that the model's predict method returns values within a specific range."""
    # This is a dummy test; in real scenarios, you'd check for expected values
    dummy_input = tf.random.uniform([1, 10])
    prediction = model.predict(dummy_input)
    assert (prediction >= 0).all() and (prediction <= 1).all(), "Prediction values are out of expected range."