"""
LIME explainers for ClarivueXAI.

This module provides explainers that use LIME (Local Interpretable Model-agnostic Explanations)
to explain model predictions.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseExplainer, BaseModel, ExplanationResult
from clarivuexai.core.registry import register_explainer
from clarivuexai.core.utils import get_feature_names


@register_explainer('lime')
class LimeExplainer(BaseExplainer):
    """
    Explainer that uses LIME to explain model predictions.
    
    This explainer uses the LIME library to generate local explanations
    for model predictions.
    """
    
    def __init__(self, model: BaseModel, mode: str = 'tabular', **kwargs):
        """
        Initialize a LimeExplainer.
        
        Args:
            model: A ClarivueXAI model wrapper
            mode: Type of data ('tabular', 'text', or 'image')
            **kwargs: Additional arguments for the LIME explainer
        """
        try:
            import lime
        except ImportError:
            raise ImportError("LIME is required to use LimeExplainer. "
                             "Install it with 'pip install lime'.")
        
        super().__init__(model, 'lime')
        self.mode = mode
        self.kwargs = kwargs
        self._explainer = None
        
    def _create_explainer(self, X: Union[np.ndarray, pd.DataFrame, list]) -> None:
        """
        Create a LIME explainer based on the data type.
        
        Args:
            X: Input data (can be numpy array, pandas DataFrame, or list of strings for text)
        """
        try:
            import lime
            import lime.lime_tabular
            import lime.lime_text
            import lime.lime_image
        except ImportError:
            raise ImportError("LIME is required to use LimeExplainer. "
                            "Install it with 'pip install lime'.")
        
        # Auto-detect the mode if not explicitly set
        if self.mode == 'auto':
            if isinstance(X, list) and all(isinstance(item, str) for item in X):
                self.mode = 'text'
            elif isinstance(X, np.ndarray) and len(X.shape) > 2:
                # Multi-dimensional arrays are likely images
                self.mode = 'image'
            else:
                # Default to tabular for all other cases
                self.mode = 'tabular'
        
        # Get feature names if possible
        feature_names = self.model.feature_names
        
        # For text data, we don't need to get traditional feature names
        if self.mode != 'text' and feature_names is None:
            try:
                feature_names = get_feature_names(X)
            except TypeError:
                # If we can't get feature names, use generic ones
                if isinstance(X, np.ndarray):
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                elif isinstance(X, list) and X and isinstance(X[0], (list, np.ndarray)):
                    feature_names = [f"feature_{i}" for i in range(len(X[0]))]
                else:
                    # Default case - create a single generic feature name
                    feature_names = ["feature_0"]
        
        # Create the appropriate LIME explainer based on the data type
        if self.mode == 'tabular':
            # For tabular data
            categorical_features = self.kwargs.get('categorical_features', [])
            
            # Convert X to numpy array if it's not already
            if isinstance(X, pd.DataFrame):
                training_data = X.values
            elif isinstance(X, list):
                try:
                    training_data = np.array(X)
                except:
                    # If conversion fails, use a small random placeholder
                    training_data = np.random.random((10, len(feature_names)))
            else:
                training_data = X
            
            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                class_names=self.kwargs.get('class_names', None),
                categorical_features=categorical_features,
                categorical_names=self.kwargs.get('categorical_names', None),
                kernel_width=self.kwargs.get('kernel_width', 3),
                verbose=self.kwargs.get('verbose', False),
                mode=self.kwargs.get('lime_mode', 'classification' if self.model.is_classifier else 'regression')
            )
        elif self.mode == 'text':
            # For text data
            self._explainer = lime.lime_text.LimeTextExplainer(
                class_names=self.kwargs.get('class_names', ['negative', 'positive']),
                kernel_width=self.kwargs.get('kernel_width', 25),
                verbose=self.kwargs.get('verbose', False)
            )
        elif self.mode == 'image':
            # For image data
            self._explainer = lime.lime_image.LimeImageExplainer(
                kernel_width=self.kwargs.get('kernel_width', 0.25),
                verbose=self.kwargs.get('verbose', False)
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
    def _map_lime_feature_to_original(self, lime_feature: str, feature_names: List[str]) -> str:
        """
        Map a LIME feature name back to the original feature name.
        
        LIME often returns feature names with additions like "feature > value" or other formatting.
        This method extracts the original feature name.
        
        Args:
            lime_feature: Feature name returned by LIME
            feature_names: List of original feature names
            
        Returns:
            Original feature name
        """
        # Ensure feature_names is a list, not a pandas Index
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()
        
        # First try direct match
        if lime_feature in feature_names:
            return lime_feature
        
        # Try to find the original feature name that is contained in the LIME feature
        for original_name in feature_names:
            if original_name in lime_feature:
                return original_name
        
        # If that fails, try extracting from common LIME patterns
        # Pattern: "feature > value" or "feature < value"
        if " > " in lime_feature or " < " in lime_feature:
            parts = lime_feature.split(" > ") if " > " in lime_feature else lime_feature.split(" < ")
            potential_feature = parts[0].strip()
            if potential_feature in feature_names:
                return potential_feature
        
        # Pattern: "feature [category]"
        if " [" in lime_feature and "]" in lime_feature:
            potential_feature = lime_feature.split(" [")[0].strip()
            if potential_feature in feature_names:
                return potential_feature
        
        # Last resort - try to find the closest matching feature name
        best_match = None
        best_match_len = 0
        for original_name in feature_names:
            if original_name in lime_feature and len(original_name) > best_match_len:
                best_match = original_name
                best_match_len = len(original_name)
        
        if best_match:
            return best_match
        
        # If we can't find a match, raise an error
        raise ValueError(f"Could not map LIME feature '{lime_feature}' to any original feature in {feature_names}")
    
    def explain_global(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> ExplanationResult:
        """
        Generate global explanations for the model using LIME.
        
        LIME is primarily a local explanation method, so this method
        generates local explanations for a sample of instances and
        aggregates them to create a global explanation.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Returns:
            Global explanation results
        """
        # Create explainer if not already created
        if self._explainer is None:
            self._create_explainer(X)
        
        # Get feature names
        feature_names = kwargs.get('feature_names', None)
        if feature_names is None:
            feature_names = self.model.feature_names
        if feature_names is None:
            feature_names = get_feature_names(X)
        
        # Convert pandas Index to list if needed
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()
        
        # Sample instances for explanation
        n_samples = kwargs.get('n_samples', min(10, len(X)))
        if isinstance(X, pd.DataFrame):
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            sample = X.iloc[sample_indices]
        else:
            sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
            sample = X[sample_indices]
        
        # Generate local explanations for each sample
        local_explanations = []
        for i in range(n_samples):
            if self.mode == 'tabular':
                instance = sample.iloc[i] if isinstance(sample, pd.DataFrame) else sample[i]
                explanation = self._explainer.explain_instance(
                    instance,
                    self.model.predict_proba if self.model.is_classifier else self.model.predict,
                    num_features=len(feature_names),
                    **{k: v for k, v in kwargs.items() if k != 'mode'}  # Remove 'mode' from kwargs
                )
                local_explanations.append(explanation)
            elif self.mode == 'text':
                instance = sample.iloc[i] if isinstance(sample, pd.DataFrame) else sample[i]
                explanation = self._explainer.explain_instance(
                    instance,
                    self.model.predict_proba if self.model.is_classifier else self.model.predict,
                    num_features=kwargs.get('num_features', 10),
                    **{k: v for k, v in kwargs.items() if k != 'mode'}  # Remove 'mode' from kwargs
                )
                local_explanations.append(explanation)
            elif self.mode == 'image':
                instance = sample.iloc[i] if isinstance(sample, pd.DataFrame) else sample[i]
                explanation = self._explainer.explain_instance(
                    instance,
                    self.model.predict_proba if self.model.is_classifier else self.model.predict,
                    **{k: v for k, v in kwargs.items() if k != 'mode'}  # Remove 'mode' from kwargs
                )
                local_explanations.append(explanation)
        
        # Aggregate local explanations to create a global explanation
        feature_importance = np.zeros(len(feature_names))
        for explanation in local_explanations:
            if self.mode == 'tabular':
                for feature, importance in explanation.as_list():
                    try:
                        # Use the mapping function to find the original feature
                        original_feature = self._map_lime_feature_to_original(feature, feature_names)
                        feature_idx = feature_names.index(original_feature)
                        feature_importance[feature_idx] += abs(importance)
                    except (ValueError, IndexError) as e:
                        # Log the error but continue processing
                        print(f"Warning: Could not map feature '{feature}': {str(e)}")
                        continue
            elif self.mode == 'text':
                for word, importance in explanation.as_list():
                    if word in feature_names:
                        feature_idx = feature_names.index(word)
                        feature_importance[feature_idx] += abs(importance)
            elif self.mode == 'image':
                # For image explanations, we don't have a straightforward way to aggregate
                # Just store the explanations for visualization
                pass
        
        # Normalize feature importance
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        # Create explanation data
        explanation_data = {
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'local_explanations': local_explanations
        }
        
        return ExplanationResult(
            explanation_type='global',
            data=explanation_data,
            feature_names=feature_names,
            explainer_name=self.name
        )
    
    def explain_local(self, X: Union[np.ndarray, pd.DataFrame, list], **kwargs) -> ExplanationResult:
        """
        Generate local explanations for specific instances using LIME.
        
        Args:
            X: Input data (can be numpy array, pandas DataFrame, or list of strings for text)
            **kwargs: Additional arguments
            
        Returns:
            Local explanation results
        """
        # Handle setting mode from kwargs
        if 'mode' in kwargs:
            self.mode = kwargs['mode']
            self._explainer = None  # Force recreation of the explainer
        
        # Handle text data specifically
        if isinstance(X, list) and all(isinstance(item, str) for item in X):
            # For text data, set the mode to 'text' if not already set
            if self.mode != 'text':
                self.mode = 'text'
                self._explainer = None  # Force recreation of the explainer
        
        # Create explainer if not already created
        if self._explainer is None:
            self._create_explainer(X)
        
        # Get feature names
        feature_names = kwargs.get('feature_names', None)
        if feature_names is None:
            feature_names = self.model.feature_names
        if feature_names is None:
            try:
                feature_names = get_feature_names(X)
            except:
                # For text data, we might not have predefined feature names
                if self.mode == 'text':
                    # Will be populated later with actual words
                    feature_names = []
                else:
                    # Default to generic feature names
                    feature_names = [f"feature_{i}" for i in range(10)]
        
        # Convert pandas Index to list if needed
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()
        
        # Remove 'mode' from kwargs before passing to LIME methods
        lime_kwargs = {k: v for k, v in kwargs.items() if k != 'mode'}
        
        # Initialize explanation variable to avoid UnboundLocalError
        explanation = None
        feature_importance = None
        
        # Generate local explanation based on the data type
        if self.mode == 'tabular':
            if isinstance(X, pd.DataFrame):
                instance = X.iloc[0]
            else:
                instance = X[0]
            explanation = self._explainer.explain_instance(
                instance,
                self.model.predict_proba if self.model.is_classifier else self.model.predict,
                num_features=len(feature_names) if hasattr(instance, '__len__') else lime_kwargs.get('num_features', 10),
                **lime_kwargs
            )
            
            # Extract feature importance
            feature_importance = np.zeros(len(feature_names))
            for feature, importance in explanation.as_list():
                try:
                    # Use the mapping function to find the original feature
                    original_feature = self._map_lime_feature_to_original(feature, feature_names)
                    feature_idx = feature_names.index(original_feature)
                    feature_importance[feature_idx] = importance
                except (ValueError, IndexError) as e:
                    # Log the error but continue processing
                    print(f"Warning: Could not map feature '{feature}': {str(e)}")
                    continue
        
        elif self.mode == 'text':
            # For text data, we process the first text sample
            instance = X[0]  # Just take the first text
            
            try:
                import lime.lime_text
                # For text, we need a different approach to get the explainer
                if self._explainer is None:
                    self._explainer = lime.lime_text.LimeTextExplainer(
                        class_names=lime_kwargs.get('class_names', ['negative', 'positive']),
                        kernel_width=lime_kwargs.get('kernel_width', 25),
                        verbose=lime_kwargs.get('verbose', False)
                    )
                
                # For text classification, we typically use predict_proba
                explanation = self._explainer.explain_instance(
                    instance,
                    self.model.predict_proba,
                    num_features=lime_kwargs.get('num_features', 10),
                    **lime_kwargs
                )
                
                # For text, we don't map to predefined features - each word is a feature
                # Just return the words and their importance scores
                words = []
                importance_scores = []
                for word, importance in explanation.as_list():
                    words.append(word)
                    importance_scores.append(importance)
                
                # Create a placeholder feature importance array
                feature_importance = np.array(importance_scores)
                # Update feature names to be the actual words
                feature_names = words
                
            except Exception as e:
                print(f"Error generating text explanation: {str(e)}")
                # Create empty placeholder data
                feature_importance = np.array([])
                feature_names = []
        
        elif self.mode == 'image':
            instance = X.iloc[0] if isinstance(X, pd.DataFrame) and len(X) == 1 else X[0]
            explanation = self._explainer.explain_instance(
                instance,
                self.model.predict_proba if self.model.is_classifier else self.model.predict,
                **lime_kwargs
            )
            
            # For image explanations, we don't have a straightforward feature importance
            # but we still need to provide something
            feature_importance = np.array([])
        
        # Ensure we have an explanation or create a placeholder
        if explanation is None:
            print("Warning: Could not generate a valid explanation")
            if feature_importance is None:
                feature_importance = np.array([])
        
        # Create explanation data
        explanation_data = {
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'explanation': explanation,
            'mode': self.mode  # Add mode to help with visualization
        }
        
        return ExplanationResult(
            explanation_type='local',
            data=explanation_data,
            feature_names=feature_names,
            explainer_name=self.name
        )