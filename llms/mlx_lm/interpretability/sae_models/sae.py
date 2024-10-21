import mlx.nn as nn
import mlx.core as mx
from abc import ABC, abstractmethod

class SAE(nn.Module, ABC):
    def __init__(self, d_model: int, d_features: int):
        super().__init__()
        self.d_model = d_model
        self.d_features = d_features

    @abstractmethod
    def encode(self, x: mx.array) -> mx.array:
        ...
    
    @abstractmethod
    def decode(self, x: mx.array) -> mx.array:
        ...
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.decode(self.encode(x))
    
    def reconstruction_error(self, x: mx.array) -> mx.array:
        return x - self(x)