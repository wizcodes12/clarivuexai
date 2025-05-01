"""
Tests for the core module of ClarivueXAI.

This module contains tests for the core components of ClarivueXAI,
including base classes and registry.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseModel, BaseExplainer, ExplanationResult
from clarivuexai.core.registry import Registry, register_model, register_explainer, register_data_handler
from clarivuexai.core.utils import (
    get_feature_names, check_model_compatibility, detect_framework,
    convert_to_numpy, is_classifier, get_model_metadata
)


class TestBaseModel(unittest.TestCase):
    """Tests for the BaseModel class."""
    
    def test_init(self):
        """Test initialization of BaseModel."""
        # Create a mock model
        mock_model = MagicMock()
        
        # Create a subclass of BaseModel for testing
        class TestModel(BaseModel):
            def predict(self, X):
                return np.zeros(len(X))
            
            def predict_proba(self, X):
                return np.zeros((len(X), 2))
        
        # Initialize the model
        model = TestModel(mock_model, 'classifier', ['feature1', 'feature2'])
        
        # Check attributes
        self.assertEqual(model.model, mock_model)
        self.assertEqual(model.model_type, 'classifier')
        self.assertEqual(model.feature_names, ['feature1', 'feature2'])


class TestBaseExplainer(unittest.TestCase):
    """Tests for the BaseExplainer class."""
    
    def test_init(self):
        """Test initialization of BaseExplainer."""
        # Create a mock model
        mock_model = MagicMock(spec=BaseModel)
        
        # Create a subclass of BaseExplainer for testing
        class TestExplainer(BaseExplainer):
            def explain_global(self, X, **kwargs):
                return ExplanationResult('global', {}, ['feature1', 'feature2'], 'test')
            
            def explain_local(self, X, **kwargs):
                return ExplanationResult('local', {}, ['feature1', 'feature2'], 'test')
        
        # Initialize the explainer
        explainer = TestExplainer(mock_model, 'test_explainer')
        
        # Check attributes
        self.assertEqual(explainer.model, mock_model)
        self.assertEqual(explainer.name, 'test_explainer')


class TestExplanationResult(unittest.TestCase):
    """Tests for the ExplanationResult class."""
    
    def test_init(self):
        """Test initialization of ExplanationResult."""
        # Create an explanation result
        explanation = ExplanationResult(
            explanation_type='global',
            data={'importances': np.array([0.5, 0.3, 0.2])},
            feature_names=['feature1', 'feature2', 'feature3'],
            explainer_name='test_explainer'
        )
        
        # Check attributes
        self.assertEqual(explanation.explanation_type, 'global')
        self.assertEqual(explanation.feature_names, ['feature1', 'feature2', 'feature3'])
        self.assertEqual(explanation.explainer_name, 'test_explainer')
        np.testing.assert_array_equal(explanation.data['importances'], np.array([0.5, 0.3, 0.2]))
    
    def test_to_dict(self):
        """Test to_dict method of ExplanationResult."""
        # Create an explanation result
        explanation = ExplanationResult(
            explanation_type='global',
            data={'importances': np.array([0.5, 0.3, 0.2])},
            feature_names=['feature1', 'feature2', 'feature3'],
            explainer_name='test_explainer'
        )
        
        # Convert to dictionary
        explanation_dict = explanation.to_dict()
        
        # Check dictionary
        self.assertEqual(explanation_dict['explanation_type'], 'global')
        self.assertEqual(explanation_dict['feature_names'], ['feature1', 'feature2', 'feature3'])
        self.assertEqual(explanation_dict['explainer_name'], 'test_explainer')
        np.testing.assert_array_equal(explanation_dict['data']['importances'], np.array([0.5, 0.3, 0.2]))


class TestRegistry(unittest.TestCase):
    """Tests for the Registry class."""
    
    def setUp(self):
        """Set up the test case."""
        self.registry = Registry()
        
        # Create mock classes
        class MockModel(BaseModel):
            def predict(self, X):
                return np.zeros(len(X))
            
            def predict_proba(self, X):
                return np.zeros((len(X), 2))
        
        class MockExplainer(BaseExplainer):
            def explain_global(self, X, **kwargs):
                return ExplanationResult('global', {}, ['feature1', 'feature2'], 'test')
            
            def explain_local(self, X, **kwargs):
                return ExplanationResult('local', {}, ['feature1', 'feature2'], 'test')
        
        class MockDataHandler:
            pass
        
        self.MockModel = MockModel
        self.MockExplainer = MockExplainer
        self.MockDataHandler = MockDataHandler
    
    def test_register_model(self):
        """Test registering a model."""
        # Register a model
        self.registry.register_model('mock_model', self.MockModel)
        
        # Check if the model is registered
        self.assertIn('mock_model', self.registry.list_models())
        self.assertEqual(self.registry.get_model('mock_model'), self.MockModel)
    
    def test_register_explainer(self):
        """Test registering an explainer."""
        # Register an explainer
        self.registry.register_explainer('mock_explainer', self.MockExplainer)
        
        # Check if the explainer is registered
        self.assertIn('mock_explainer', self.registry.list_explainers())
        self.assertEqual(self.registry.get_explainer('mock_explainer'), self.MockExplainer)
    
    def test_register_data_handler(self):
        """Test registering a data handler."""
        # Register a data handler
        with self.assertRaises(TypeError):
            self.registry.register_data_handler('mock_handler', self.MockDataHandler)


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""
    
    def test_get_feature_names(self):
        """Test get_feature_names function."""
        # Test with DataFrame
        df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        self.assertEqual(get_feature_names(df), ['feature1', 'feature2'])
        
        # Test with numpy array
        X = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(get_feature_names(X), ['feature_0', 'feature_1', 'feature_2'])
    
    def test_convert_to_numpy(self):
        """Test convert_to_numpy function."""
        # Test with DataFrame
        df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        np.testing.assert_array_equal(convert_to_numpy(df), np.array([[1, 4], [2, 5], [3, 6]]))
        
        # Test with numpy array
        X = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(convert_to_numpy(X), X)
        
        # Test with list
        X_list = [[1, 2, 3], [4, 5, 6]]
        np.testing.assert_array_equal(convert_to_numpy(X_list), np.array(X_list))


if __name__ == '__main__':
    unittest.main()