"""
Interactive visualization tools for ClarivueXAI.

This module provides functions for creating modern, interactive visualizations
with a clean and aesthetic design for explanation results.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

def create_interactive_plot(
    explanation: ExplanationResult,
    plot_type: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create an interactive plot with a modern design for the given explanation.

    Args:
        explanation: Explanation result to visualize
        plot_type: Type of plot to create
        **kwargs: Additional arguments for the plot

    Returns:
        Plotly figure
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError("Plotly is required for interactive plotting. "
                         "Install it with 'pip install plotly'.")

    # Determine plot type
    if plot_type is None:
        plot_type = 'feature_importance' if explanation.explanation_type == 'global' else 'local_explanation'

    # Create the appropriate plot
    plot_functions = {
        'feature_importance': interactive_feature_importance,
        'local_explanation': interactive_local_explanation,
        'shap_summary': interactive_shap_summary,
        'counterfactual': interactive_counterfactual,
        'integrated_gradients': interactive_integrated_gradients,
    }

    if plot_type not in plot_functions:
        raise ValueError(f"Unknown plot type: {plot_type}")

    return plot_functions[plot_type](explanation, **kwargs)


def interactive_feature_importance(
    explanation: ExplanationResult,
    top_n: int = 10,
    sort: bool = True,
    **kwargs
) -> Any:
    """
    Create an interactive feature importance plot with a modern design.

    Args:
        explanation: Explanation result to visualize
        top_n: Number of top features to show
        sort: Whether to sort features by importance
        **kwargs: Additional arguments for the plot

    Returns:
        Plotly figure
    """
    # Get feature importance
    importance = None
    if 'feature_importance' in explanation.data:
        importance = explanation.data['feature_importance']
    elif 'importances' in explanation.data:
        importance = explanation.data['importances']
    elif 'shap_values' in explanation.data:
        importance = np.mean(np.abs(explanation.data['shap_values']), axis=0)
    else:
        raise ValueError("Explanation does not contain feature importance data")

    feature_names = explanation.feature_names

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    # Sort and limit
    if sort:
        df = df.sort_values('Importance', ascending=False)
    if top_n and top_n < len(df):
        df = df.head(top_n)

    # Create figure
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
            text=kwargs.get('title', f'Feature Importance ({explanation.explainer_name})'),
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=COLORS['primary']),
        ),
        xaxis_title=kwargs.get('xlabel', 'Importance'),
        yaxis_title=kwargs.get('ylabel', 'Feature'),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=kwargs.get('height', 600),
        width=kwargs.get('width', 800),
        margin=dict(l=150, r=50, t=80, b=50),
    )

    return fig


def interactive_local_explanation(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Create an interactive local explanation plot with a modern design.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the plot

    Returns:
        Plotly figure
    """
    if 'shap_values' in explanation.data:
        return interactive_shap_local(explanation, **kwargs)
    elif 'attributions' in explanation.data:
        return interactive_integrated_gradients(explanation, **kwargs)
    elif 'counterfactual' in explanation.data:
        return interactive_counterfactual(explanation, **kwargs)
    else:
        return interactive_feature_importance(explanation, **kwargs)


def interactive_shap_local(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Create an interactive local SHAP explanation plot with a modern design.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the plot

    Returns:
        Plotly figure
    """
    shap_values = explanation.data['shap_values']
    feature_names = explanation.feature_names
    
    # Handle different shapes of SHAP values
    if len(shap_values.shape) > 2:
        # For multi-class models, take the class with highest absolute SHAP values
        # or the class specified in kwargs
        class_idx = kwargs.get('class_idx', np.argmax(np.sum(np.abs(shap_values), axis=(1, 2))))
        shap_vals = shap_values[class_idx, 0, :]  # Take first instance for the selected class
    elif len(shap_values.shape) == 2:
        # For binary/regression models
        shap_vals = shap_values[0, :]  # Take first instance
    else:
        # Single instance, single class
        shap_vals = shap_values
    
    # Ensure the shape is compatible - extract to 1D array if needed
    if hasattr(shap_vals, 'shape') and len(shap_vals.shape) > 1:
        shap_vals = shap_vals.flatten()
    
    # Ensure the feature names and SHAP values have matching lengths
    if len(feature_names) != len(shap_vals):
        print(f"Warning: Feature names length ({len(feature_names)}) does not match shap_vals length ({len(shap_vals)})")
        # Truncate to the shorter length
        min_len = min(len(feature_names), len(shap_vals))
        feature_names = feature_names[:min_len]
        shap_vals = shap_vals[:min_len]

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_vals
    })

    # Sort by absolute SHAP value
    df['Abs SHAP'] = np.abs(df['SHAP Value'])
    df = df.sort_values('Abs SHAP', ascending=False)

    # Limit to top N
    top_n = kwargs.get('top_n', 10)
    if top_n and top_n < len(df):
        df = df.head(top_n)

    # Create figure
    fig = go.Figure()
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in df['SHAP Value']]

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
            text=kwargs.get('title', 'SHAP Values'),
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=COLORS['primary']),
        ),
        xaxis_title=kwargs.get('xlabel', 'SHAP Value'),
        yaxis_title=kwargs.get('ylabel', 'Feature'),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=kwargs.get('height', 600),
        width=kwargs.get('width', 800),
        margin=dict(l=150, r=50, t=80, b=50),
    )

    return fig


def interactive_shap_summary(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Create an interactive SHAP summary plot with a modern design.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the plot

    Returns:
        Plotly figure
    """
    shap_values = explanation.data['shap_values']
    feature_names = explanation.feature_names
    
    # Handle different shapes of SHAP values
    if len(shap_values.shape) > 2:
        # For multi-class models with shape (n_classes, n_samples, n_features)
        # Average across classes and samples
        mean_abs_shap = np.mean(np.mean(np.abs(shap_values), axis=0), axis=0)
    elif len(shap_values.shape) == 2:
        # For binary/regression models with shape (n_samples, n_features)
        # Average across samples
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    else:
        # Single instance, single class
        mean_abs_shap = np.abs(shap_values)
    
    # Ensure the shape is compatible - extract to 1D array if needed
    if hasattr(mean_abs_shap, 'shape') and len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.flatten()
    
    # Ensure the feature names and SHAP values have matching lengths
    if len(feature_names) != len(mean_abs_shap):
        print(f"Warning: Feature names length ({len(feature_names)}) does not match mean_abs_shap length ({len(mean_abs_shap)})")
        # Truncate to the shorter length
        min_len = min(len(feature_names), len(mean_abs_shap))
        feature_names = feature_names[:min_len]
        mean_abs_shap = mean_abs_shap[:min_len]

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    })

    # Sort and limit
    df = df.sort_values('Mean |SHAP|', ascending=False)
    top_n = kwargs.get('top_n', 10)
    if top_n and top_n < len(df):
        df = df.head(top_n)

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
            text=kwargs.get('title', 'SHAP Summary Plot'),
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=COLORS['primary']),
        ),
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Feature',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=kwargs.get('height', 600),
        width=kwargs.get('width', 800),
        margin=dict(l=150, r=50, t=80, b=50),
    )

    return fig
def debug_interactive_plot(func):
    """
    Decorator to add debug information and error handling to interactive plotting functions.
    
    Args:
        func: The plotting function to wrap
        
    Returns:
        Wrapped function with additional debug and error handling
    """
    import functools
    import traceback
    
    @functools.wraps(func)
    def wrapper(explanation, **kwargs):
        try:
            # Print debug info about the explanation
            print(f"\nDebug info for {func.__name__}:")
            
            # Check for shap_values
            if 'shap_values' in explanation.data:
                shap_shape = explanation.data['shap_values'].shape
                print(f"SHAP values shape: {shap_shape}")
            else:
                print("No SHAP values found in explanation data")
            
            # Check feature names
            if hasattr(explanation, 'feature_names'):
                print(f"Feature names length: {len(explanation.feature_names)}")
                print(f"Sample feature names: {explanation.feature_names[:3]}...")
            else:
                print("No feature names found in explanation")
            
            # Execute the function
            result = func(explanation, **kwargs)
            print(f"{func.__name__} completed successfully")
            return result
            
        except Exception as e:
            print(f"\nError in {func.__name__}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            
            # Return a simple error plot
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"Error creating plot: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(
                title=f"Error in {func.__name__}",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )
            return fig
            
    return wrapper




def interactive_counterfactual(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Create an interactive counterfactual explanation plot with a modern design.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the plot

    Returns:
        Plotly figure
    """
    original = explanation.data['original']
    counterfactual = explanation.data['counterfactual']
    feature_names = explanation.feature_names

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Original': original,
        'Counterfactual': counterfactual,
        'Difference': counterfactual - original
    })

    # Sort by absolute difference
    df['Abs Difference'] = np.abs(df['Difference'])
    df = df.sort_values('Abs Difference', ascending=False)

    # Limit to top N
    top_n = kwargs.get('top_n', 10)
    if top_n and top_n < len(df):
        df = df.head(top_n)

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=('Original vs Counterfactual', 'Feature Changes'),
        vertical_spacing=0.15
    )

    # Original vs Counterfactual
    fig.add_trace(
        go.Bar(
            x=df['Original'],
            y=df['Feature'],
            orientation='h',
            name='Original',
            marker=dict(
                color=COLORS['secondary'],
                line=dict(color=COLORS['primary'], width=2),
            ),
            hovertemplate='Original: %{x:.4f}<br>%{y}<extra></extra>',
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=df['Counterfactual'],
            y=df['Feature'],
            orientation='h',
            name='Counterfactual',
            marker=dict(
                color=COLORS['accent'],
                line=dict(color=COLORS['primary'], width=2),
            ),
            hovertemplate='Counterfactual: %{x:.4f}<br>%{y}<extra></extra>',
        ),
        row=1,
        col=1
    )

    # Differences
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in df['Difference']]
    fig.add_trace(
        go.Bar(
            x=df['Difference'],
            y=df['Feature'],
            orientation='h',
            name='Difference',
            marker=dict(
                color=colors,
                line=dict(color=COLORS['primary'], width=2),
            ),
            hovertemplate='Change: %{x:.4f}<br>%{y}<extra></extra>',
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        title=dict(
            text=kwargs.get('title', 'Counterfactual Explanation'),
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=COLORS['primary']),
        ),
        height=kwargs.get('height', 800),
        width=kwargs.get('width', 800),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=150, r=50, t=100, b=50),
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor=COLORS['primary'],
            borderwidth=1,
        ),
    )

    return fig


def interactive_integrated_gradients(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Create an interactive integrated gradients explanation plot with a modern design.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the plot

    Returns:
        Plotly figure
    """
    attributions = explanation.data['attributions']
    feature_names = explanation.feature_names

    # Ensure 1D attributions
    attributions = attributions[0] if len(attributions.shape) > 1 else attributions

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Attribution': attributions
    })

    # Sort by absolute attribution
    df['Abs Attribution'] = np.abs(df['Attribution'])
    df = df.sort_values('Abs Attribution', ascending=False)

    # Limit to top N
    top_n = kwargs.get('top_n', 10)
    if top_n and top_n < len(df):
        df = df.head(top_n)

    # Create figure
    fig = go.Figure()
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in df['Attribution']]

    fig.add_trace(go.Bar(
        x=df['Attribution'],
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
            text=kwargs.get('title', 'Integrated Gradients'),
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=COLORS['primary']),
        ),
        xaxis_title=kwargs.get('xlabel', 'Attribution'),
        yaxis_title=kwargs.get('ylabel', 'Feature'),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=kwargs.get('height', 600),
        width=kwargs.get('width', 800),
        margin=dict(l=150, r=50, t=80, b=50),
    )

    return fig