"""
Registry for ClarivueXAI components.

This module provides a registry system for models, explainers, and data handlers
to enable dynamic discovery and instantiation of components.
"""

from typing import Any, Dict, List, Optional, Type, Union

from clarivuexai.core.base import BaseDataHandler, BaseExplainer, BaseModel


class Registry:
    """
    Registry for ClarivueXAI components.
    
    This class maintains registries of available models, explainers, and data handlers,
    allowing for dynamic discovery and instantiation of components.
    """
    
    def __init__(self):
        """Initialize the registry with empty dictionaries."""
        self._models: Dict[str, Type[BaseModel]] = {}
        self._explainers: Dict[str, Type[BaseExplainer]] = {}
        self._data_handlers: Dict[str, Type[BaseDataHandler]] = {}
        
    def register_model(self, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class.
        
        Args:
            name: Name to register the model under
            model_class: Model class to register
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Expected subclass of BaseModel, got {model_class}")
        self._models[name] = model_class
        
    def register_explainer(self, name: str, explainer_class: Type[BaseExplainer]) -> None:
        """
        Register an explainer class.
        
        Args:
            name: Name to register the explainer under
            explainer_class: Explainer class to register
        """
        if not issubclass(explainer_class, BaseExplainer):
            raise TypeError(f"Expected subclass of BaseExplainer, got {explainer_class}")
        self._explainers[name] = explainer_class
        
    def register_data_handler(self, name: str, handler_class: Type[BaseDataHandler]) -> None:
        """
        Register a data handler class.
        
        Args:
            name: Name to register the data handler under
            handler_class: Data handler class to register
        """
        if not issubclass(handler_class, BaseDataHandler):
            raise TypeError(f"Expected subclass of BaseDataHandler, got {handler_class}")
        self._data_handlers[name] = handler_class
        
    def get_model(self, name: str) -> Type[BaseModel]:
        """
        Get a registered model class by name.
        
        Args:
            name: Name of the model class
            
        Returns:
            Registered model class
            
        Raises:
            KeyError: If no model is registered under the given name
        """
        if name not in self._models:
            raise KeyError(f"No model registered under name '{name}'")
        return self._models[name]
    
    def get_explainer(self, name: str) -> Type[BaseExplainer]:
        """
        Get a registered explainer class by name.
        
        Args:
            name: Name of the explainer class
            
        Returns:
            Registered explainer class
            
        Raises:
            KeyError: If no explainer is registered under the given name
        """
        if name not in self._explainers:
            raise KeyError(f"No explainer registered under name '{name}'")
        return self._explainers[name]
    
    def get_data_handler(self, name: str) -> Type[BaseDataHandler]:
        """
        Get a registered data handler class by name.
        
        Args:
            name: Name of the data handler class
            
        Returns:
            Registered data handler class
            
        Raises:
            KeyError: If no data handler is registered under the given name
        """
        if name not in self._data_handlers:
            raise KeyError(f"No data handler registered under name '{name}'")
        return self._data_handlers[name]
    
    def list_models(self) -> List[str]:
        """
        List all registered model names.
        
        Returns:
            List of registered model names
        """
        return list(self._models.keys())
    
    def list_explainers(self) -> List[str]:
        """
        List all registered explainer names.
        
        Returns:
            List of registered explainer names
        """
        return list(self._explainers.keys())
    
    def list_data_handlers(self) -> List[str]:
        """
        List all registered data handler names.
        
        Returns:
            List of registered data handler names
        """
        return list(self._data_handlers.keys())


# Create a global registry instance
registry = Registry()


def register_model(name: str):
    """
    Decorator for registering model classes.
    
    Args:
        name: Name to register the model under
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        registry.register_model(name, cls)
        return cls
    return decorator


def register_explainer(name: str):
    """
    Decorator for registering explainer classes.
    
    Args:
        name: Name to register the explainer under
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        registry.register_explainer(name, cls)
        return cls
    return decorator


def register_data_handler(name: str):
    """
    Decorator for registering data handler classes.
    
    Args:
        name: Name to register the data handler under
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        registry.register_data_handler(name, cls)
        return cls
    return decorator