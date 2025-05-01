"""
Dashboard creation tools for ClarivueXAI.

This module provides functions for creating modern, interactive dashboards
with a clean and aesthetic design for exploring explanation results.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from clarivuexai.core.base import ExplanationResult

# Modern color palette
COLORS = {
    'primary': '#2E3B55',  # Dark blue-gray
    'secondary': '#00A3E0',  # Bright cyan
    'accent': '#FF6B6B',  # Coral
    'background': '#F5F7FA',  # Light gray
    'text': '#333333',  # Dark gray
    'positive': '#4CAF50',  # Green
    'negative': '#F44336',  # Red
}

# CSS styles for consistent design
STYLES = {
    'container': {
        'padding': '20px',
        'backgroundColor': COLORS['background'],
        'fontFamily': 'Arial, sans-serif',
        'color': COLORS['text'],
    },
    'header': {
        'color': COLORS['primary'],
        'fontWeight': 'bold',
        'marginBottom': '20px',
    },
    'card': {
        'backgroundColor': 'white',
        'padding': '15px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '20px',
    },
    'slider': {
        'margin': '20px 0',
    },
}

def create_dashboard(
    explanation: ExplanationResult,
    dashboard_type: str = 'basic',
    **kwargs
) -> Any:
    """
    Create an interactive dashboard for the given explanation.

    Args:
        explanation: Explanation result to visualize
        dashboard_type: Type of dashboard to create ('basic', 'shap', 'comparison')
        **kwargs: Additional arguments for the dashboard

    Returns:
        Dash app object
    """
    if dashboard_type == 'basic':
        return create_basic_dashboard(explanation, **kwargs)
    elif dashboard_type == 'shap':
        return create_shap_dashboard(explanation, **kwargs)
    elif dashboard_type == 'comparison':
        return create_comparison_dashboard(explanation, **kwargs)
    else:
        raise ValueError(f"Unknown dashboard type: {dashboard_type}")


def create_basic_dashboard(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Create a basic dashboard with a modern design for the given explanation.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the dashboard

    Returns:
        Dash app object
    """
    try:
        import dash
        from dash import dcc, html
    except ImportError:
        raise ImportError("Dash and Plotly are required for dashboards. "
                         "Install them with 'pip install dash plotly'.")

    # Initialize Dash app with external stylesheets for modern look
    app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

    # Get feature importance
    importance = None
    if 'feature_importance' in explanation.data:
        importance = explanation.data['feature_importance']
    elif 'importances' in explanation.data:
        importance = explanation.data['importances']
    elif 'shap_values' in explanation.data:
        importance = np.mean(np.abs(explanation.data['shap_values']), axis=0)
    feature_names = explanation.feature_names

    # Create layout
    app.layout = html.Div(
        style=STYLES['container'],
        children=[
            html.H1(
                f"ClarivueXAI Dashboard: {explanation.explainer_name}",
                style=STYLES['header']
            ),
            html.Div(
                style=STYLES['card'],
                children=[
                    html.H3("Explanation Type", style={'color': COLORS['primary']}),
                    html.P(explanation.explanation_type.capitalize(), style={'fontSize': '16px'}),
                ]
            ),
            html.Div(
                style=STYLES['card'],
                children=[
                    html.H3("Feature Importance", style={'color': COLORS['primary']}),
                    dcc.Graph(
                        id='feature-importance',
                        figure=create_feature_importance_figure(feature_names, importance),
                        config={'displayModeBar': False}
                    ),
                ]
            ),
            html.Div(
                style=STYLES['card'],
                children=[
                    html.H3("Settings", style={'color': COLORS['primary']}),
                    html.Label("Number of features to display:", style={'fontSize': '14px'}),
                    dcc.Slider(
                        id='n-features-slider',
                        min=5,
                        max=min(50, len(feature_names)),
                        step=5,
                        value=10,
                        marks={i: str(i) for i in range(5, min(50, len(feature_names)) + 1, 5)},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                    ),
                ]
            ),
        ]
    )

    # Callback to update graph
    @app.callback(
        Output('feature-importance', 'figure'),
        [Input('n-features-slider', 'value')]
    )
    def update_graph(n_features):
        return create_feature_importance_figure(feature_names, importance, n_features)

    return app


def create_feature_importance_figure(
    feature_names: List[str],
    importance: np.ndarray,
    n_features: int = 10
) -> Any:
    """
    Create a feature importance figure' figure with a modern design.

    Args:
        feature_names: List of feature names
        importance: Feature importance values
        n_features: Number of features to display

    Returns:
        Plotly figure
    """
    if importance is None:
        return go.Figure()

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    # Sort by importance
    df = df.sort_values('Importance', ascending=False).head(n_features)

    # Create figure with modern styling
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker=dict(
            color=COLORS['secondary'],
            line=dict(color=COLORS['primary'], width=2),
        ),
        hovertemplate='%{x:.4f}<br>%{y}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text='Feature Importance',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=COLORS['primary']),
        ),
        xaxis_title='Importance',
        yaxis_title='Feature',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=150, r=50, t=80, b=50),
        showlegend=False,
        height=600,
    )

    return fig


def create_shap_dashboard(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Create a SHAP dashboard with a modern design for the given explanation.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the dashboard

    Returns:
        Dash app object
    """
    try:
        import dash
        from dash import dcc, html
    except ImportError:
        raise ImportError("Dash and Plotly are required for dashboards. "
                         "Install them with 'pip install dash plotly'.")

    # Check if SHAP values are available
    if 'shap_values' not in explanation.data:
        raise ValueError("Explanation does not contain SHAP values")

    shap_values = explanation.data['shap_values']
    feature_names = explanation.feature_names

    # Initialize Dash app
    app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

    # Create layout
    app.layout = html.Div(
        style=STYLES['container'],
        children=[
            html.H1("SHAP Dashboard", style=STYLES['header']),
            html.Div(
                style=STYLES['card'],
                children=[
                    html.H3("SHAP Summary", style={'color': COLORS['primary']}),
                    dcc.Graph(
                        id='shap-summary',
                        figure=create_shap_summary_figure(shap_values, feature_names),
                        config={'displayModeBar': False}
                    ),
                ]
            ),
            html.Div(
                style=STYLES['card'],
                children=[
                    html.H3("SHAP Values", style={'color': COLORS['primary']}),
                    dcc.Dropdown(
                        id='instance-dropdown',
                        options=[{'label': f'Instance {i}', 'value': i} for i in range(len(shap_values))],
                        value=0,
                        style={'width': '50%', 'marginBottom': '20px'},
                    ),
                    dcc.Graph(id='shap-values', config={'displayModeBar': False}),
                ]
            ),
            html.Div(
                style=STYLES['card'],
                children=[
                    html.H3("Settings", style={'color': COLORS['primary']}),
                    html.Label("Number of features to display:", style={'fontSize': '14px'}),
                    dcc.Slider(
                        id='n-features-slider',
                        min=5,
                        max=min(50, len(feature_names)),
                        step=5,
                        value=10,
                        marks={i: str(i) for i in range(5, min(50, len(feature_names)) + 1, 5)},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                    ),
                ]
            ),
        ]
    )

    # Callbacks
    @app.callback(
        Output('shap-summary', 'figure'),
        [Input('n-features-slider', 'value')]
    )
    def update_summary(n_features):
        return create_shap_summary_figure(shap_values, feature_names, n_features)

    @app.callback(
        Output('shap-values', 'figure'),
        [Input('instance-dropdown', 'value'), Input('n-features-slider', 'value')]
    )
    def update_shap_values(instance_idx, n_features):
        return create_shap_values_figure(shap_values, feature_names, instance_idx, n_features)

    return app


def create_shap_summary_figure(
    shap_values: np.ndarray,
    feature_names: List[str],
    n_features: int = 10
) -> Any:
    """
    Create a SHAP summary figure with a modern design.

    Args:
        shap_values: SHAP values
        feature_names: List of feature names
        n_features: Number of features to display

    Returns:
        Plotly figure
    """
    # Compute mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    })

    # Sort and limit
    df = df.sort_values('Mean |SHAP|', ascending=False).head(n_features)

    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Mean |SHAP|'],
        y=df['Feature'],
        orientation='h',
        marker=dict(
            color=COLORS['secondary'],
            line=dict(color=COLORS['primary'], width=2),
        ),
        hovertemplate='%{x:.4f}<br>%{y}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text='SHAP Summary',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=COLORS['primary']),
        ),
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Feature',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=150, r=50, t=80, b=50),
        height=600,
    )

    return fig


def create_shap_values_figure(
    shap_values: np.ndarray,
    feature_names: List[str],
    instance_idx: int = 0,
    n_features: int = 10
) -> Any:
    """
    Create a SHAP values figure for a specific instance with a modern design.

    Args:
        shap_values: SHAP values
        feature_names: List of feature names
        instance_idx: Index of the instance to visualize
        n_features: Number of features to display

    Returns:
        Plotly figure
    """
    # Get SHAP values for the instance
    instance_shap = shap_values[instance_idx]

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': instance_shap
    })

    # Sort by absolute SHAP value
    df['Abs SHAP'] = np.abs(df['SHAP Value'])
    df = df.sort_values('Abs SHAP', ascending=False).head(n_features)

    # Create figure
    fig = go.Figure()
    colors = [COLORS['positive'] if val > 0 else COLORS['negative'] for val in df['SHAP Value']]

    fig.add_trace(go.Bar(
        x=df['SHAP Value'],
        y=df['Feature'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color=COLORS['primary'], width=2),
        ),
        hovertemplate='%{x:.4f}<br>%{y}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text=f'SHAP Values for Instance {instance_idx}',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=COLORS['primary']),
        ),
        xaxis_title='SHAP Value',
        yaxis_title='Feature',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=150, r=50, t=80, b=50),
        height=600,
    )

    return fig


def create_comparison_dashboard(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Create a comparison dashboard for multiple models with a modern design.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the dashboard

    Returns:
        Dash app object
    """
    try:
        import dash
        from dash import dcc, html
    except ImportError:
        raise ImportError("Dash and Plotly are required for dashboards. "
                         "Install them with 'pip install dash plotly'.")

    # Check comparison data
    if 'comparisons' not in explanation.data:
        raise ValueError("Explanation does not contain comparison data")

    comparisons = explanation.data['comparisons']
    feature_names = explanation.feature_names
    model_names = list(comparisons.keys())

    # Initialize Dash app
    app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

    # Create layout
    app.layout = html.Div(
        style=STYLES['container'],
        children=[
            html.H1("Model Comparison Dashboard", style=STYLES['header']),
            html.Div(
                style=STYLES['card'],
                children=[
                    html.H3("Model Selection", style={'color': COLORS['primary']}),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[{'label': model, 'value': model} for model in model_names],
                        value=model_names[:2],  # Select first two models by default
                        multi=True,
                        style={'width': '100%', 'marginBottom': '20px'},
                    ),
                ]
            ),
            html.Div(
                style=STYLES['card'],
                children=[
                    html.H3("Feature Importance Comparison", style={'color': COLORS['primary']}),
                    dcc.Graph(id='comparison-graph', config={'displayModeBar': False}),
                ]
            ),
            html.Div(
                style=STYLES['card'],
                children=[
                    html.H3("Settings", style={'color': COLORS['primary']}),
                    html.Label("Number of features to display:", style={'fontSize': '14px'}),
                    dcc.Slider(
                        id='n-features-slider',
                        min=5,
                        max=min(50, len(feature_names)),
                        step=5,
                        value=10,
                        marks={i: str(i) for i in range(5, min(50, len(feature_names)) + 1, 5)},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                    ),
                ]
            ),
        ]
    )

    # Callback to update comparison graph
    @app.callback(
        Output('comparison-graph', 'figure'),
        [Input('model-dropdown', 'value'), Input('n-features-slider', 'value')]
    )
    def update_comparison_graph(selected_models, n_features):
        return create_comparison_figure(comparisons, feature_names, selected_models, n_features)

    return app


def create_comparison_figure(
    comparisons: Dict[str, np.ndarray],
    feature_names: List[str],
    selected_models: Union[str, List[str]],
    n_features: int = 10
) -> Any:
    """
    Create a comparison figure for multiple models with a modern design.

    Args:
        comparisons: Dictionary mapping model names to importance values
        feature_names: List of feature names
        selected_models: Model(s) to include in the comparison
        n_features: Number of features to display

    Returns:
        Plotly figure
    """
    # Handle single model case
    if isinstance(selected_models, str):
        selected_models = [selected_models]

    # Create figure
    fig = go.Figure()

    # Get union of top features
    all_features = set()
    for model in selected_models:
        if model in comparisons:
            importance = comparisons[model]
            top_indices = np.argsort(importance)[-n_features:]
            for idx in top_indices:
                all_features.add(feature_names[idx])

    # Sort features by average importance
    feature_avg_importance = {}
    for feat in all_features:
        feat_idx = feature_names.index(feat)
        total = sum(comparisons[model][feat_idx] for model in selected_models if model in comparisons)
        count = sum(1 for model in selected_models if model in comparisons)
        feature_avg_importance[feat] = total / count if count > 0 else 0

    sorted_features = sorted(all_features, key=lambda x: feature_avg_importance[x], reverse=True)[:n_features]

    # Color palette for models
    model_colors = [COLORS['secondary'], COLORS['accent'], COLORS['positive'], COLORS['negative']]

    # Add traces for each model
    for idx, model in enumerate(selected_models):
        if model in comparisons:
            importance = comparisons[model]
            feat_importance = [importance[feature_names.index(feat)] for feat in sorted_features]
            fig.add_trace(go.Bar(
                y=sorted_features,
                x=feat_importance,
                name=model,
                orientation='h',
                marker=dict(
                    color=model_colors[idx % len(model_colors)],
                    line=dict(color=COLORS['primary'], width=2),
                ),
                hovertemplate=f'{model}<br>%{{x:.4f}}<br>%{{y}}<extra></extra>',
            ))

    fig.update_layout(
        title=dict(
            text='Feature Importance Comparison',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=COLORS['primary']),
        ),
        xaxis_title='Importance',
        yaxis_title='Feature',
        barmode='group',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=150, r=50, t=80, b=50),
        height=600,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor=COLORS['primary'],
            borderwidth=1,
        ),
    )

    return fig