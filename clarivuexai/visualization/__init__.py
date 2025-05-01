"""
Visualization tools for ClarivueXAI.

This module contains tools for visualizing explanation results with modern,
interactive, and static plotting functions.
"""

# Import from plots.py
from clarivuexai.visualization.plots import (
    create_plot,
    plot_feature_importance,
    plot_local_explanation,
    plot_shap_local,
    plot_shap_summary,
    plot_shap_dependence,
    plot_lime_explanation,
    plot_counterfactual,
    plot_integrated_gradients
)

# Import from interactive.py
from clarivuexai.visualization.interactive import (
    create_interactive_plot,
    interactive_feature_importance,
    interactive_local_explanation,
    interactive_shap_local,
    interactive_shap_summary,
    interactive_counterfactual,
    interactive_integrated_gradients
)

# Import from dashboards.py
from clarivuexai.visualization.dashboards import (
    create_dashboard,
    create_basic_dashboard,
    create_shap_dashboard,
    create_comparison_dashboard
)

__all__ = [
    # From plots.py
    'create_plot',
    'plot_feature_importance',
    'plot_local_explanation',
    'plot_shap_local',
    'plot_shap_summary',
    'plot_shap_dependence',
    'plot_lime_explanation',
    'plot_counterfactual',
    'plot_integrated_gradients',
    
    # From interactive.py
    'create_interactive_plot',
    'interactive_feature_importance',
    'interactive_local_explanation',
    'interactive_shap_local',
    'interactive_shap_summary',
    'interactive_counterfactual',
    'interactive_integrated_gradients',
    'debug_interactive_plot',
    
    # From dashboards.py
    'create_dashboard',
    'create_basic_dashboard',
    'create_shap_dashboard',
    'create_comparison_dashboard'
]