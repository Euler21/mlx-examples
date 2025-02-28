from typing import Dict
import mlx.core as mx
import mlx.nn as nn

class ActivationCache:
    """
    Stores activations from a model run.
    """
    def __init__(
        self, 
        activations: Dict[str, mx.array] = {}, 
        model_name: str | None = None, 
        prompt: str | None = None
    ):
        self.activations = activations
        self.model_name = model_name
        self.prompt = prompt

    def set(self, name: str, activations: mx.array):
        self.activations[name] = activations

    def get(self, name: str) -> mx.array | None:
        return self.activations.get(name)

    def clear(self):
        """
        Clear the activation cache.
        """
        self.activations = {}
        self.prompt = None
        self.model_name = None

    def prepare_for_prompt(self, prompt: str, model_name: str | None = None):
        """Set up the activation cache to run on a given prompt. 
        
        Stores the prompt and clears the cache.
        """
        self.clear()
        self.model_name = model_name
        self.prompt = prompt

    def save_safetensors(self, file: str):
        """Saves the activations to a specified file in safetensors format.
        """
        metadata = None
        if self.model_name or self.prompt:
            metadata = {}
            if self.model_name:
                metadata['model_name'] = self.model_name
            if self.prompt:
                metadata['prompt'] = self.prompt
        mx.save_safetensors(file, self.activations, metadata)

    def load_safetensors(self, file: str):
        """Loads cached activations and metadata into this instance from a safetensors file.
        """
        self.activations, metadata = mx.load(
            file, 
            format="safetensors", 
            return_metadata=True
        )
        if metadata:
            self.prompt = metadata.get("prompt", None)
            self.model_name = metadata.get("model_name", None)

    @classmethod
    def from_safetensors(cls, file: str) -> 'ActivationCache':
        """Creates a new ActivationCache instance from a safetensors file.
        """
        cache = cls()
        cache.load_safetensors(file)
        return cache

    def remove_batch_dim(self):
        for key in self.activations:
            assert (
                 self.activations[key].shape[0] == 1
            ), f"Cannot remove batch dimension from cache with batch size > 1, \
                for key {key} with shape {self.activations[key].shape}"
            self.activations[key] = self.activations[key][0]

    def run_prompt(
        self,
        model: nn.Module, 
        tokenizer, 
        prompt: str, 
        model_name: str | None = None,
    ):
        self.prepare_for_prompt(prompt, model_name)
        inputs = mx.array(tokenizer.encode(prompt))[None]
        model(inputs)[0]
        self.remove_batch_dim()