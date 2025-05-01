"""
Core module for ClarivueXAI.

This module contains the core components of ClarivueXAI,
including base classes, registry, and utility functions.
"""

from clarivuexai.core.base import BaseModel, BaseExplainer, ExplanationResult, BaseDataHandler
from clarivuexai.core.registry import registry, register_model, register_explainer, register_data_handler
from clarivuexai.core.utils import (
    get_feature_names, check_model_compatibility, detect_framework,
    convert_to_numpy, is_classifier, get_model_metadata,
    # Additional utility functions
    check_array, check_is_fitted
)