"""
Tests for the visualization module of ClarivueXAI.

This module contains tests for the visualization tools in ClarivueXAI,
including plotting functions and interactive visualizations.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from clarivuexai.core.base import ExplanationResult
from clarivuexai.visualization.plots import create_plot, plot_feature_importance


class TestPlots(unittest.TestCase):
    """Tests for the plotting functions."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a mock explanation result
        self.explanation = ExplanationResult(
            explanation_type='global',
            data={
                'importances': np.array([0.5, 0.3, 0.2]),
                'feature_names': ['feature1', 'feature2', 'feature3']
            },
            feature_names=['feature1', 'feature2', 'feature3'],
            explainer_name='test_explainer'
        )
    
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.barplot')
    def test_plot_feature_importance(self, mock_barplot, mock_figure):
        """Test plot_feature_importance function."""
        # Create a plot
        plot = plot_feature_importance(self.explanation)
        
        # Check that the plot was created
        mock_figure.assert_called_once()
        mock_barplot.assert_called_once()
    
    @patch('clarivuexai.visualization.plots.plot_feature_importance')
    def test_create_plot_feature_importance(self, mock_plot_feature_importance):
        """Test create_plot function with feature_importance plot type."""
        # Create a plot
        plot = create_plot(self.explanation, plot_type='feature_importance')
        
        # Check that the plot was created
        mock_plot_feature_importance.assert_called_once_with(self.explanation)
    
    @patch('clarivuexai.visualization.plots.plot_local_explanation')
    def test_create_plot_local_explanation(self, mock_plot_local_explanation):
        """Test create_plot function with local_explanation plot type."""
        # Create a local explanation
        local_explanation = ExplanationResult(
            explanation_type='local',
            data={
                'feature_importance': np.array([0.5, 0.3, 0.2]),
                'feature_names': ['feature1', 'feature2', 'feature3']
            },
            feature_names=['feature1', 'feature2', 'feature3'],
            explainer_name='test_explainer'
        )
        
        # Create a plot
        plot = create_plot(local_explanation, plot_type='local_explanation')
        
        # Check that the plot was created
        mock_plot_local_explanation.assert_called_once_with(local_explanation)


if __name__ == '__main__':
    unittest.main()