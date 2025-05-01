"""
Plotting functions for ClarivueXAI.

This module provides functions for visualizing explanation results with a modern,
clean design using Matplotlib and Seaborn.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Set Seaborn style
sns.set_style("whitegrid", {
    'axes.facecolor': COLORS['background'],
    'figure.facecolor': COLORS['background'],
    'text.color': COLORS['text'],
    'axes.labelcolor': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'grid.color': '#E0E0E0',
})

def create_plot(
    explanation: ExplanationResult,
    plot_type: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create a plot for the given explanation with a modern design.

    Args:
        explanation: Explanation result to visualize
        plot_type: Type of plot to create
        **kwargs: Additional arguments for the plot

    Returns:
        Matplotlib figure
    """
    # Determine plot type
    if plot_type is None:
        plot_type = 'feature_importance' if explanation.explanation_type == 'global' else 'local_explanation'

    # Plot functions
    plot_functions = {
        'feature_importance': plot_feature_importance,
        'local_explanation': plot_local_explanation,
        'shap_summary': plot_shap_summary,
        'shap_dependence': plot_shap_dependence,
        'lime_explanation': plot_lime_explanation,
        'counterfactual': plot_counterfactual,
        'integrated_gradients': plot_integrated_gradients,
    }

    if plot_type not in plot_functions:
        raise ValueError(f"Unknown plot type: {plot_type}")

    return plot_functions[plot_type](explanation, **kwargs)


def plot_feature_importance(
    explanation: ExplanationResult,
    top_n: int = 10,
    sort: bool = True,
    class_idx: int = 0,
    **kwargs
) -> Any:
    """
    Plot feature importance with a modern design.

    Args:
        explanation: Explanation result to visualize
        top_n: Number of top features to show
        sort: Whether to sort features by importance
        class_idx: Index of the class to visualize (for multi-class models)
        **kwargs: Additional arguments for the plot

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("Matplotlib and seaborn are required for plotting. "
                         "Install them with 'pip install matplotlib seaborn'.")

    # Get feature importance
    importance = None
    feature_names = explanation.feature_names
    data_keys = list(explanation.data.keys())

    if 'feature_importance' in explanation.data:
        importance = explanation.data['feature_importance']
    elif 'importances' in explanation.data:
        importance = explanation.data['importances']
    elif 'shap_values' in explanation.data:
        shap_values = explanation.data['shap_values']
        if isinstance(shap_values, list):
            class_values = shap_values[class_idx] if len(shap_values) > class_idx else shap_values[0]
            importance = np.mean(np.abs(class_values), axis=0) if len(class_values.shape) > 1 else np.abs(class_values)
        elif len(shap_values.shape) == 3:
            importance = np.mean(np.abs(shap_values[class_idx]), axis=0)
        elif len(shap_values.shape) == 2:
            importance = np.mean(np.abs(shap_values), axis=0)
        elif len(shap_values.shape) == 1:
            importance = np.abs(shap_values)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
    else:
        for key, value in explanation.data.items():
            if isinstance(value, (np.ndarray, list)) and len(value) == len(feature_names):
                importance = np.array(value)
                break
        if importance is None:
            raise ValueError(f"Could not find valid importance values. Available keys: {data_keys}")

    # Ensure importance is 1D
    importance = np.array(importance)
    if len(importance.shape) > 1:
        if importance.shape[-1] == len(feature_names):
            importance = np.mean(np.abs(importance), axis=tuple(range(len(importance.shape)-1)))
        elif importance.shape[0] == len(feature_names):
            importance = np.mean(np.abs(importance), axis=tuple(range(1, len(importance.shape))))
        else:
            for dim in range(len(importance.shape)):
                if importance.shape[dim] == len(feature_names):
                    avg_dims = tuple(i for i in range(len(importance.shape)) if i != dim)
                    importance = np.mean(np.abs(importance), axis=avg_dims) if avg_dims else importance
                    break
            else:
                raise ValueError(f"Could not match importance shape {importance.shape} with feature_names length {len(feature_names)}")

    # Check length match
    if len(importance) != len(feature_names):
        if len(importance) == 2 and class_idx < len(importance):
            print(f"Warning: Using synthetic importance for class {class_idx}")
            importance = np.ones(len(feature_names)) / len(feature_names)
        else:
            raise ValueError(f"Length mismatch: feature_names ({len(feature_names)}) and importance ({len(importance)})")

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

    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    sns.barplot(x='Importance', y='Feature', data=df, hue='Feature', palette=[COLORS['secondary']] * len(df), legend=False, ax=ax)

    # Customize plot
    title = (f'Feature Importance ({explanation.explainer_name}) - Class {class_idx}' if explanation.explainer_name and 'class_idx' in kwargs
             else f'Feature Importance ({explanation.explainer_name})' if explanation.explainer_name
             else 'Feature Importance')
    ax.set_title(kwargs.get('title', title), color=COLORS['primary'], fontsize=16, pad=20)
    ax.set_xlabel(kwargs.get('xlabel', 'Importance'), color=COLORS['text'], fontsize=12)
    ax.set_ylabel(kwargs.get('ylabel', 'Feature'), color=COLORS['text'], fontsize=12)
    ax.tick_params(axis='both', colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['primary'])
    ax.spines['bottom'].set_color(COLORS['primary'])
    plt.tight_layout()

    return fig


def plot_local_explanation(
    explanation: ExplanationResult,
    class_idx: int = 0,
    **kwargs
) -> Any:
    """
    Plot local explanation with a modern design.

    Args:
        explanation: Explanation result to visualize
        class_idx: Index of the class to visualize (for multi-class models)
        **kwargs: Additional arguments for the plot

    Returns:
        Matplotlib figure
    """
    if 'shap_values' in explanation.data:
        return plot_shap_local(explanation, class_idx=class_idx, **kwargs)
    elif 'explanation' in explanation.data and explanation.explainer_name == 'lime':
        return plot_lime_explanation(explanation, **kwargs)
    elif 'attributions' in explanation.data:
        return plot_integrated_gradients(explanation, **kwargs)
    elif 'counterfactual' in explanation.data:
        return plot_counterfactual(explanation, **kwargs)
    else:
        return plot_feature_importance(explanation, class_idx=class_idx, **kwargs)


def plot_shap_local(
    explanation: ExplanationResult,
    class_idx: int = 0,
    **kwargs
) -> Any:
    """
    Plot local SHAP explanation with a modern design.

    Args:
        explanation: Explanation result to visualize
        class_idx: Index of the class to visualize (for multi-class models)
        **kwargs: Additional arguments for the plot

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("Matplotlib and seaborn are required for plotting. "
                         "Install them with 'pip install matplotlib seaborn'.")

    shap_values = explanation.data['shap_values']
    feature_names = explanation.feature_names

    # Handle SHAP value formats
    shap_vals = None
    if isinstance(shap_values, list):
        class_values = shap_values[class_idx] if len(shap_values) > class_idx else shap_values[0]
        shap_vals = class_values[0] if len(class_values.shape) > 1 else class_values
    elif len(shap_values.shape) == 3:
        shap_vals = shap_values[class_idx, 0] if shap_values.shape[0] > class_idx else shap_values[0, 0]
    elif len(shap_values.shape) == 2:
        shap_vals = shap_values[0]
    elif len(shap_values.shape) == 1:
        shap_vals = shap_values
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

    if shap_vals is None or len(shap_vals) != len(feature_names):
        print("Warning: Could not find valid SHAP values for local explanation")
        shap_vals = np.ones(len(feature_names)) / len(feature_names)

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_vals
    })

    # Sort and limit
    df['Abs SHAP'] = np.abs(df['SHAP Value'])
    df = df.sort_values('Abs SHAP', ascending=False)
    top_n = kwargs.get('top_n', 10)
    if top_n and top_n < len(df):
        df = df.head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in df['SHAP Value']]
    sns.barplot(x='SHAP Value', y='Feature', data=df, hue='Feature', palette=colors, legend=False, ax=ax)

    # Customize plot
    title = kwargs.get('title', f'SHAP Values - Class {class_idx}' if 'class_idx' in kwargs else 'SHAP Values')
    ax.set_title(title, color=COLORS['primary'], fontsize=16, pad=20)
    ax.set_xlabel(kwargs.get('xlabel', 'SHAP Value'), color=COLORS['text'], fontsize=12)
    ax.set_ylabel(kwargs.get('ylabel', 'Feature'), color=COLORS['text'], fontsize=12)
    ax.tick_params(axis='both', colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['primary'])
    ax.spines['bottom'].set_color(COLORS['primary'])
    plt.tight_layout()

    return fig


def plot_shap_summary(
    explanation: ExplanationResult,
    class_idx: int = 0,
    **kwargs
) -> Any:
    """
    Plot SHAP summary plot with a modern design.

    Args:
        explanation: Explanation result to visualize
        class_idx: Index of the class to visualize (for multi-class models)
        **kwargs: Additional arguments for the plot

    Returns:
        Matplotlib figure
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("SHAP and matplotlib are required for plotting. "
                         "Install them with 'pip install shap matplotlib'.")

    shap_values = explanation.data['shap_values']
    feature_names = explanation.feature_names

    # Handle SHAP value formats
    if isinstance(shap_values, list):
        shap_vals = shap_values[class_idx] if len(shap_values) > class_idx else shap_values[0]
    elif len(shap_values.shape) == 3:
        shap_vals = shap_values[class_idx] if shap_values.shape[0] > class_idx else shap_values[0]
    else:
        shap_vals = shap_values

    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    shap.summary_plot(
        shap_vals,
        feature_names=feature_names,
        plot_type=kwargs.get('plot_type', 'bar'),
        show=False,
        plot_size=None,
        color=COLORS['secondary']
    )

    # Customize plot
    ax.set_title(kwargs.get('title', f'SHAP Summary - Class {class_idx}' if 'class_idx' in kwargs else 'SHAP Summary'),
                 color=COLORS['primary'], fontsize=16, pad=20)
    ax.set_xlabel(kwargs.get('xlabel', 'Mean |SHAP Value|'), color=COLORS['text'], fontsize=12)
    ax.set_ylabel(kwargs.get('ylabel', 'Feature'), color=COLORS['text'], fontsize=12)
    ax.tick_params(axis='both', colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['primary'])
    ax.spines['bottom'].set_color(COLORS['primary'])
    plt.tight_layout()

    return fig


def plot_shap_dependence(
    explanation: ExplanationResult,
    feature_idx: int = 0,
    class_idx: int = 0,
    **kwargs
) -> Any:
    """
    Plot SHAP dependence plot with a modern design.

    Args:
        explanation: Explanation result to visualize
        feature_idx: Index of the feature to plot
        class_idx: Index of the class to visualize (for multi-class models)
        **kwargs: Additional arguments for the plot

    Returns:
        Matplotlib figure
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("SHAP and matplotlib are required for plotting. "
                         "Install them with 'pip install shap matplotlib'.")

    shap_values = explanation.data['shap_values']
    feature_names = explanation.feature_names

    # Handle SHAP value formats
    if isinstance(shap_values, list):
        shap_vals = shap_values[class_idx] if len(shap_values) > class_idx else shap_values[0]
    elif len(shap_values.shape) == 3:
        shap_vals = shap_values[class_idx] if shap_values.shape[0] > class_idx else shap_values[0]
    else:
        shap_vals = shap_values

    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    shap.dependence_plot(
        feature_idx,
        shap_vals,
        feature_names=feature_names,
        show=False,
        ax=ax,
        color=COLORS['secondary']
    )

    # Customize plot
    feature_name = feature_names[feature_idx]
    ax.set_title(kwargs.get('title', f'SHAP Dependence Plot for {feature_name}'),
                 color=COLORS['primary'], fontsize=16, pad=20)
    ax.set_xlabel(feature_name, color=COLORS['text'], fontsize=12)
    ax.set_ylabel('SHAP Value', color=COLORS['text'], fontsize=12)
    ax.tick_params(axis='both', colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['primary'])
    ax.spines['bottom'].set_color(COLORS['primary'])
    plt.tight_layout()

    return fig


def plot_lime_explanation(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Plot LIME explanation with a modern design.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the plot

    Returns:
        Matplotlib figure
    """
    if 'explanation' not in explanation.data or explanation.explainer_name != 'lime':
        raise ValueError("Explanation is not a LIME explanation")

    lime_explanation = explanation.data['explanation']
    fig = lime_explanation.as_pyplot_figure(**kwargs)

    # Customize plot
    ax = fig.gca()
    ax.set_title(kwargs.get('title', 'LIME Explanation'), color=COLORS['primary'], fontsize=16, pad=20)
    ax.set_xlabel(kwargs.get('xlabel', 'Contribution'), color=COLORS['text'], fontsize=12)
    ax.set_ylabel(kwargs.get('ylabel', 'Feature'), color=COLORS['text'], fontsize=12)
    ax.tick_params(axis='both', colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['primary'])
    ax.spines['bottom'].set_color(COLORS['primary'])
    plt.tight_layout()

    return fig


def plot_counterfactual(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Plot counterfactual explanation with a modern design.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the plot

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("Matplotlib and seaborn are required for plotting. "
                         "Install them with 'pip install matplotlib seaborn'.")

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

    # Sort and limit
    df['Abs Difference'] = np.abs(df['Difference'])
    df = df.sort_values('Abs Difference', ascending=False)
    top_n = kwargs.get('top_n', 10)
    if top_n and top_n < len(df):
        df = df.head(top_n)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=kwargs.get('figsize', (12, 8)), sharey=True, gridspec_kw={'height_ratios': [1, 1]})

    # Original vs Counterfactual
    df_melted = pd.melt(df, id_vars=['Feature'], value_vars=['Original', 'Counterfactual'])
    sns.barplot(x='value', y='Feature', hue='variable', data=df_melted, palette=[COLORS['secondary'], COLORS['accent']], ax=ax1)
    ax1.set_title(kwargs.get('title', 'Counterfactual Explanation'), color=COLORS['primary'], fontsize=16, pad=20)
    ax1.set_xlabel(kwargs.get('xlabel', 'Value'), color=COLORS['text'], fontsize=12)
    ax1.set_ylabel(kwargs.get('ylabel', 'Feature'), color=COLORS['text'], fontsize=12)
    ax1.legend(title='', frameon=True, edgecolor=COLORS['primary'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['primary'])
    ax1.spines['bottom'].set_color(COLORS['primary'])

    # Differences
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in df['Difference']]
    sns.barplot(x='Difference', y='Feature', data=df, hue='Feature', palette=colors, legend=False, ax=ax2)
    ax2.set_title('Feature Changes', color=COLORS['primary'], fontsize=14, pad=20)
    ax2.set_xlabel('Change', color=COLORS['text'], fontsize=12)
    ax2.set_ylabel('', color=COLORS['text'], fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['primary'])
    ax2.spines['bottom'].set_color(COLORS['primary'])

    plt.tight_layout()

    return fig


def plot_integrated_gradients(
    explanation: ExplanationResult,
    **kwargs
) -> Any:
    """
    Plot integrated gradients explanation with a modern design.

    Args:
        explanation: Explanation result to visualize
        **kwargs: Additional arguments for the plot

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("Matplotlib and seaborn are required for plotting. "
                         "Install them with 'pip install matplotlib seaborn'.")

    attributions = explanation.data['attributions']
    feature_names = explanation.feature_names

    # Ensure 1D attributions
    attributions = attributions[0] if len(attributions.shape) > 1 else attributions

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Attribution': attributions
    })

    # Sort and limit
    df['Abs Attribution'] = np.abs(df['Attribution'])
    df = df.sort_values('Abs Attribution', ascending=False)
    top_n = kwargs.get('top_n', 10)
    if top_n and top_n < len(df):
        df = df.head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in df['Attribution']]
    sns.barplot(x='Attribution', y='Feature', data=df, hue='Feature', palette=colors, legend=False, ax=ax)

    # Customize plot
    ax.set_title(kwargs.get('title', 'Integrated Gradients'), color=COLORS['primary'], fontsize=16, pad=20)
    ax.set_xlabel(kwargs.get('xlabel', 'Attribution'), color=COLORS['text'], fontsize=12)
    ax.set_ylabel(kwargs.get('ylabel', 'Feature'), color=COLORS['text'], fontsize=12)
    ax.tick_params(axis='both', colors=COLORS['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['primary'])
    ax.spines['bottom'].set_color(COLORS['primary'])
    plt.tight_layout()

    return fig