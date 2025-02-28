import mlx.nn as nn
import mlx.core as mx

from typing import Any, Type, TypeVar, Dict, List, override
from activation_cache import ActivationCache
from utils import process_module_paths


class PatchedLayer(nn.Module):
    def __init__(self, name: str, layer: nn.Module, cache: ActivationCache | None):
        if isinstance(layer, PatchedLayer):
            # Avoid double patching
            layer = layer["layer"]
        self["layer"] = layer
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "activation_cache", cache)

    def __repr__(self):
        return f"({type(self).__name__}) {repr(self["layer"])}"
    
    def __call__(self, *args, **kwargs):
        activations = self["layer"](*args, **kwargs)
        self.cache(activations)
        return self.modify(activations)
    
    def __getattr__(self, attr):
        return getattr(self["layer"], attr)
    
    def __setattr__(self, attr, value):
        setattr(self["layer"], attr, value)

    def __delattr__(self, attr):
        delattr(self["layer"], attr)

    def cache(self, activations):
        name: str = object.__getattribute__(self, "name")
        cache: ActivationCache | None = object.__getattribute__(self, "activation_cache")
        if cache:
            cache.set(name, activations)

    def modify(self, activations):
        return activations

class SteerablePatchedLayer(PatchedLayer):
    def __init__(
        self, 
        name: str, 
        layer: nn.Module, 
        direction: mx.array, 
        strength: float,
        cache: ActivationCache | None, 
    ):
        super().__init__(name, layer, cache)
        self.direction = direction
        self.strength = strength

    @override
    def modify(self, activations):
        return activations + self.strength * self.direction

def patch_layers(model: nn.Module, layers: List[int] = None, patcher = PatchedLayer):
    if layers == None:
        layers = range(len(model.layers))
    if type(patcher) is not list:
        for i in layers:
            model.layers[i] = patcher(model.layers[i])
    else:
        for i, p in zip(layers, patcher):
            model.layers[i] = p(model.layers[i])

######################
# Patching Functions #
######################

P = TypeVar('P', bound='PatchedLayer')

def patch_module(
    module: nn.Module, 
    path: str, 
    patcher_cls: Type[P], 
    *patcher_args: Any, 
    **patcher_kwargs: Any
) -> None:
    """
    Patches a nested module using dot notation with wildcard support.
    
    Args:
        module: The parent module to patch
        path: Dot-separated path to the target module (e.g., 'layers.*.mlp', 'layers.*')
        patcher_cls: Class type that constructs a patched version when instantiated with a module
        *patcher_args: Additional positional arguments to pass to the patcher constructor
        **patcher_kwargs: Additional keyword arguments to pass to the patcher constructor
        
    The patcher_cls will be called with the path as the first argument, followed by the module
    and any additional arguments. Original modules are stored in the "layer" key by patcher_cls.
    """
    def _patch_processor(container: Dict | List, key: Any, curr_path: List[str]) -> None:
        path_str = '.'.join(curr_path)
        original_module = container[key]
        patched_module = patcher_cls(path_str, original_module, *patcher_args, **patcher_kwargs)
        container[key] = patched_module
    
    process_module_paths(module, path, _patch_processor)


def unpatch_module(
    module: nn.Module,
    path: str
) -> None:
    """
    Unpatches a previously patched nested module using dot notation with wildcard support.
    
    Args:
        module: The parent module to unpatch
        path: Dot-separated path to the target module (e.g., 'layers.*.mlp', 'layers.*')
    
    This function will restore the original modules that were stored in the "layer" key
    during patching.
    """
    def _unpatch_processor(container: Dict | list, key: str | int, curr_path: List[str]) -> None:
        patched_module = container[key]
        
        # Check if this is a patched module with a "layer" key
        if isinstance(patched_module, dict) and "layer" in patched_module:
            original_module = patched_module["layer"]
            container[key] = original_module
    
    process_module_paths(module, path, _unpatch_processor)