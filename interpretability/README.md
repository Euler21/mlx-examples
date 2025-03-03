# Interpretability
Tooling for working with model internals, including activation patching, steering, and SAEs.

### Setup
```bash
pip install -e .
```

### CLI
Compute and store activations for any component of any model via the CLI:
```bash
mlx_interp.generate --prompt "Your prompt here" --model mlx-community/gemma-2-2b --activations-path "model.layers.*.mlp" --activations-output-file activations.safetensors
```

### Usage

```python
import mlx.core as mx
from mlx_lm import load

from mlx_interp.activation_cache import ActivationCache
from mlx_interp.patching import PatchedLayer, SteerablePatchedLayer, patch_module, unpatch_module

cache = ActivationCache()
model, tokenizer = load("mlx-community/gemma-2-2b")
patch_module(model, "model.layers.*", PatchedLayer, cache)
cache.run_prompt(model, tokenizer, "An interesting animal is", "gemma-2-2B")
cache.activations["model.model.layers.0"] # Read the activation at a given layer
```
