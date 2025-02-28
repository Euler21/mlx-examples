# Interpretability
Tooling for working with model internals, including activation patching, steering, and SAEs.

### Setup
TODO

### Example

```python
import mlx.core as mx
from mlx_lm import load, generate

from activation_cache import ActivationCache
from patching import PatchedLayer, SteerablePatchedLayer, patch_module, unpatch_module

cache = ActivationCache()
model, tokenizer = load("mlx-community/gemma-2-2b")
patch_module(model, "model.layers.*", PatchedLayer, cache)
cache.run_prompt(model, tokenizer, "An interesting animal is", "gemma-2-2B")
cache.activations["model.model.layers.0"] # Read the activation at a given layer
```
