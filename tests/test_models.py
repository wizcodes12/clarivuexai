"""
Tests for the model wrappers of ClarivueXAI.

This module contains tests for the model wrappers in ClarivueXAI,
including scikit-learn, TensorFlow, PyTorch, and custom models.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from clarivuexai.models.sklearn_models import SklearnModel
from clarivuexai.models.tensorflow_models import TensorflowModel
from clarivuexai.models.pytorch_models import PytorchModel
from clarivuexai.models.custom_models import CustomModel


class TestSklearnModel(unittest.TestCase):
    """Tests for the SklearnModel class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a mock scikit-learn model
        self.mock_model = MagicMock()
        self.mock_model.fit = MagicMock()
        self.mock_model.predict = MagicMock(return_value=np.array([0, 1, 0]))
        self.mock_model.predict_proba = MagicMock(return_value=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        self.mock_model._estimator_type = 'classifier'
        self.mock_model.classes_ = np.array([0, 1])
        self.mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        # Create feature names
        self.feature_names = ['feature1', 'feature2', 'feature3']
    
    def test_init(self):
        """Test initialization of SklearnModel."""
        # Initialize the model
        model = SklearnModel(self.mock_model, self.feature_names)
        
        # Check attributes
        self.assertEqual(model.model, self.mock_model)
        self.assertEqual(model.model_type, 'classifier')
        self.assertEqual(model.feature_names, self.feature_names)
        self.assertEqual(model.framework, 'sklearn')
        self.assertTrue(model.is_classifier)
        np.testing.assert_array_equal(model.classes_, np.array([0, 1]))
    
    def test_predict(self):
        """Test predict method of SklearnModel."""
        # Initialize the model
        model = SklearnModel(self.mock_model, self.feature_names)
        
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check predictions
        np.testing.assert_array_equal(predictions, np.array([0, 1, 0]))
        self.mock_model.predict.assert_called_once()
    
    def test_predict_proba(self):
        """Test predict_proba method of SklearnModel."""
        # Initialize the model
        model = SklearnModel(self.mock_model, self.feature_names)
        
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Make probability predictions
        probabilities = model.predict_proba(X)
        
        # Check probabilities
        np.testing.assert_array_equal(probabilities, np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        self.mock_model.predict_proba.assert_called_once()
    
    def test_get_feature_importances(self):
        """Test get_feature_importances method of SklearnModel."""
        # Initialize the model
        model = SklearnModel(self.mock_model, self.feature_names)
        
        # Get feature importances
        importances = model.get_feature_importances()
        
        # Check importances
        np.testing.assert_array_equal(importances, np.array([0.5, 0.3, 0.2]))


class TestCustomModel(unittest.TestCase):
    """Tests for the CustomModel class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create mock functions
        self.predict_fn = MagicMock(return_value=np.array([0, 1, 0]))
        self.predict_proba_fn = MagicMock(return_value=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        
        # Create a mock model
        self.mock_model = MagicMock()
        
        # Create feature names
        self.feature_names = ['feature1', 'feature2', 'feature3']
    
    def test_init(self):
        """Test initialization of CustomModel."""
        # Initialize the model
        model = CustomModel(
            self.mock_model, 
            predict_fn=self.predict_fn, 
            predict_proba_fn=self.predict_proba_fn, 
            feature_names=self.feature_names
        )
        
        # Check attributes
        self.assertEqual(model.model, self.mock_model)
        self.assertEqual(model.model_type, 'classifier')
        self.assertEqual(model.feature_names, self.feature_names)
        self.assertEqual(model.framework, 'custom')
        self.assertTrue(model.is_classifier)
    
    def test_predict(self):
        """Test predict method of CustomModel."""
        # Initialize the model
        model = CustomModel(
            self.mock_model, 
            predict_fn=self.predict_fn, 
            predict_proba_fn=self.predict_proba_fn, 
            feature_names=self.feature_names
        )
        
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check predictions
        np.testing.assert_array_equal(predictions, np.array([0, 1, 0]))
        self.predict_fn.assert_called_once()
    
    def test_predict_proba(self):
        """Test predict_proba method of CustomModel."""
        # Initialize the model
        model = CustomModel(
            self.mock_model, 
            predict_fn=self.predict_fn, 
            predict_proba_fn=self.predict_proba_fn, 
            feature_names=self.feature_names
        )
        
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Make probability predictions
        probabilities = model.predict_proba(X)
        
        # Check probabilities
        np.testing.assert_array_equal(probabilities, np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        self.predict_proba_fn.assert_called_once()


class TestTensorflowModel(unittest.TestCase):
    """Tests for the TensorflowModel class."""
    
    def setUp(self):
        """Set up the test case."""
        # Skip tests if TensorFlow is not installed
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            self.skipTest("TensorFlow not installed")
        
        # Create a mock TensorFlow model
        self.mock_model = MagicMock(spec=self.tf.keras.Model)
        self.mock_model.predict = MagicMock(return_value=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        
        # Mock the last layer with softmax activation
        last_layer = MagicMock()
        last_layer.activation = MagicMock()
        last_layer.activation.__name__ = 'softmax'
        
        # Set up the model
        self.mock_model.layers = [MagicMock(), MagicMock(), last_layer]
        self.mock_model.output_shape = (None, 2)
        
        # Create feature names
        self.feature_names = ['feature1', 'feature2', 'feature3']
    
    @patch('clarivuexai.models.tensorflow_models.isinstance')
    def test_init(self, mock_isinstance):
        """Test initialization of TensorflowModel."""
        # Mock isinstance check
        mock_isinstance.return_value = True
        
        # Initialize the model
        model = TensorflowModel(self.mock_model, self.feature_names)
        
        # Check attributes
        self.assertEqual(model.model, self.mock_model)
        self.assertEqual(model.model_type, 'classifier')
        self.assertEqual(model.feature_names, self.feature_names)
        self.assertEqual(model.framework, 'tensorflow')
        self.assertTrue(model.is_classifier)
        self.assertEqual(model.n_classes_, 2)
    
    @patch('clarivuexai.models.tensorflow_models.isinstance')
    def test_predict(self, mock_isinstance):
        """Test predict method of TensorflowModel."""
        # Mock isinstance check
        mock_isinstance.return_value = True
        
        # Initialize the model
        model = TensorflowModel(self.mock_model, self.feature_names)
        
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check predictions
        np.testing.assert_array_equal(predictions, np.array([0, 1, 0]))
        self.mock_model.predict.assert_called_once()
    
    @patch('clarivuexai.models.tensorflow_models.isinstance')
    def test_predict_proba(self, mock_isinstance):
        """Test predict_proba method of TensorflowModel."""
        # Mock isinstance check
        mock_isinstance.return_value = True
        
        # Initialize the model
        model = TensorflowModel(self.mock_model, self.feature_names)
        
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Make probability predictions
        probabilities = model.predict_proba(X)
        
        # Check probabilities
        np.testing.assert_array_equal(probabilities, np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        self.mock_model.predict.assert_called_once()


class TestPytorchModel(unittest.TestCase):
    """Tests for the PytorchModel class."""
    
    def setUp(self):
        """Set up the test case."""
        # Skip tests if PyTorch is not installed
        try:
            import torch
            self.torch = torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        
        # Create a mock PyTorch model
        self.mock_model = MagicMock(spec=self.torch.nn.Module)
        self.mock_model.eval = MagicMock()
        self.mock_model.to = MagicMock(return_value=self.mock_model)
        self.mock_model.zero_grad = MagicMock()
        
        # Mock forward pass
        self.mock_model.side_effect = lambda x: x
        
        # Create feature names
        self.feature_names = ['feature1', 'feature2', 'feature3']
    
    @patch('clarivuexai.models.pytorch_models.isinstance')
    @patch('clarivuexai.models.pytorch_models.torch.device')
    def test_init(self, mock_device, mock_isinstance):
        """Test initialization of PytorchModel."""
        # Mock isinstance check
        mock_isinstance.return_value = True
        
        # Mock device
        mock_device.return_value = "cpu"
        
        # Initialize the model
        model = PytorchModel(self.mock_model, self.feature_names)
        
        # Check attributes
        self.assertEqual(model.model, self.mock_model)
        self.assertEqual(model.model_type, 'classifier')
        self.assertEqual(model.feature_names, self.feature_names)
        self.assertEqual(model.framework, 'pytorch')
        self.assertTrue(model.is_classifier)
        
        # Check model operations
        self.mock_model.to.assert_called_once()
        self.mock_model.eval.assert_called_once()
    
    @patch('clarivuexai.models.pytorch_models.isinstance')
    @patch('clarivuexai.models.pytorch_models.torch.device')
    @patch('clarivuexai.models.pytorch_models.torch.tensor')
    @patch('clarivuexai.models.pytorch_models.torch.no_grad')
    def test_predict(self, mock_no_grad, mock_tensor, mock_device, mock_isinstance):
        """Test predict method of PytorchModel."""
        # Mock isinstance check
        mock_isinstance.return_value = True
        
        # Mock device
        mock_device.return_value = "cpu"
        
        # Mock tensor
        tensor_instance = MagicMock()
        mock_tensor.return_value = tensor_instance
        
        # Mock forward pass result
        result_tensor = MagicMock()
        result_tensor.cpu.return_value.numpy.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        self.mock_model.return_value = result_tensor
        
        # Mock context manager
        mock_context = MagicMock()
        mock_no_grad.return_value = mock_context
        
        # Initialize the model
        model = PytorchModel(self.mock_model, self.feature_names)
        
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check predictions
        np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))
        mock_tensor.assert_called_once()
        self.mock_model.assert_called_once()
    
    @patch('clarivuexai.models.pytorch_models.isinstance')
    @patch('clarivuexai.models.pytorch_models.torch.device')
    @patch('clarivuexai.models.pytorch_models.torch.tensor')
    @patch('clarivuexai.models.pytorch_models.torch.no_grad')
    @patch('clarivuexai.models.pytorch_models.F.softmax')
    def test_predict_proba(self, mock_softmax, mock_no_grad, mock_tensor, mock_device, mock_isinstance):
        """Test predict_proba method of PytorchModel."""
        # Mock isinstance check
        mock_isinstance.return_value = True
        
        # Mock device
        mock_device.return_value = "cpu"
        
        # Mock tensor
        tensor_instance = MagicMock()
        mock_tensor.return_value = tensor_instance
        
        # Mock forward pass result
        result_tensor = MagicMock()
        self.mock_model.return_value = result_tensor
        
        # Mock softmax result
        proba_tensor = MagicMock()
        proba_tensor.cpu.return_value.numpy.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        mock_softmax.return_value = proba_tensor
        
        # Mock context manager
        mock_context = MagicMock()
        mock_no_grad.return_value = mock_context
        
        # Initialize the model
        model = PytorchModel(self.mock_model, self.feature_names)
        
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Make probability predictions
        probabilities = model.predict_proba(X)
        
        # Check probabilities
        np.testing.assert_array_equal(probabilities, np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]))
        mock_tensor.assert_called_once()
        self.mock_model.assert_called_once()
        mock_softmax.assert_called_once_with(result_tensor, dim=1)


if __name__ == '__main__':
    unittest.main()