"""
Tests for the explainers of ClarivueXAI.

This module contains tests for the explainers in ClarivueXAI,
including feature importance, SHAP, LIME, counterfactual, and integrated gradients.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseModel, ExplanationResult
from clarivuexai.explainers.feature_importance import FeatureImportanceExplainer


class TestFeatureImportanceExplainer(unittest.TestCase):
    """Tests for the FeatureImportanceExplainer class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a mock model
        self.mock_model = MagicMock(spec=BaseModel)
        self.mock_model.get_feature_importances = MagicMock(return_value=np.array([0.5, 0.3, 0.2]))
        self.mock_model.feature_names = ['feature1', 'feature2', 'feature3']
        
        # Create an explainer
        self.explainer = FeatureImportanceExplainer(self.mock_model)
    
    def test_explain_global(self):
        """Test explain_global method of FeatureImportanceExplainer."""
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Get global explanation
        explanation = self.explainer.explain_global(X)
        
        # Check explanation
        self.assertEqual(explanation.explanation_type, 'global')
        self.assertEqual(explanation.feature_names, ['feature1', 'feature2', 'feature3'])
        self.assertEqual(explanation.explainer_name, 'feature_importance')
        np.testing.assert_array_equal(explanation.data['importances'], np.array([0.5, 0.3, 0.2]))
    
    def test_explain_local(self):
        """Test explain_local method of FeatureImportanceExplainer."""
        # Create input data
        X = np.array([[1, 2, 3]])
        
        # Check that explain_local raises an error
        with self.assertRaises(ValueError):
            self.explainer.explain_local(X)


class MockShapExplainer:
    """Mock SHAP explainer for testing."""
    
    def __init__(self, model):
        """Initialize the mock explainer."""
        self.model = model
    
    def shap_values(self, X):
        """Return mock SHAP values."""
        if isinstance(X, pd.DataFrame):
            return np.ones((len(X), X.shape[1]))
        else:
            return np.ones((len(X), X.shape[1]))


@patch('shap.Explainer', MockShapExplainer)
@patch('shap.TreeExplainer', MockShapExplainer)
@patch('shap.KernelExplainer', MockShapExplainer)
@patch('shap.DeepExplainer', MockShapExplainer)
class TestShapExplainer(unittest.TestCase):
    """Tests for the ShapExplainer class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a mock model
        self.mock_model = MagicMock(spec=BaseModel)
        self.mock_model.predict = MagicMock(return_value=np.array([0, 1, 0]))
        self.mock_model.predict_proba = MagicMock(return_value=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        self.mock_model.feature_names = ['feature1', 'feature2', 'feature3']
        self.mock_model.framework = 'sklearn'
        self.mock_model.is_classifier = True
        
        # Import here to avoid importing shap in the main test module
        from clarivuexai.explainers.shap_explainers import ShapExplainer
        
        # Create an explainer
        self.explainer = ShapExplainer(self.mock_model)
    
    def test_explain_global(self):
        """Test explain_global method of ShapExplainer."""
        # Create input data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Get global explanation
        explanation = self.explainer.explain_global(X)
        
        # Check explanation
        self.assertEqual(explanation.explanation_type, 'global')
        self.assertEqual(explanation.feature_names, ['feature1', 'feature2', 'feature3'])
        self.assertEqual(explanation.explainer_name, 'shap')
        self.assertIn('shap_values', explanation.data)
        self.assertIn('feature_importance', explanation.data)
    
    def test_explain_local(self):
        """Test explain_local method of ShapExplainer."""
        # Create input data
        X = np.array([[1, 2, 3]])
        
        # Get local explanation
        explanation = self.explainer.explain_local(X)
        
        # Check explanation
        self.assertEqual(explanation.explanation_type, 'local')
        self.assertEqual(explanation.feature_names, ['feature1', 'feature2', 'feature3'])
        self.assertEqual(explanation.explainer_name, 'shap')
        self.assertIn('shap_values', explanation.data)


if __name__ == '__main__':
    unittest.main()