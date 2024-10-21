from .sae import SAE
import mlx.nn as nn
import mlx.core as mx

class JumpReLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.threshold = mx.zeros(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return (x > self.threshold) * nn.relu(x)
    

class JumpReLUSAE(SAE):
    def __init__(self, d_model: int, d_features: int):
        super().__init__(d_model, d_features)
        self.enc = nn.Linear(d_model, d_features)
        self.dec = nn.Linear(d_features, d_model)
        self.jrelu = JumpReLU(d_features)

    def encode(self, x: mx.array) -> mx.array:
        return self.jrelu(self.enc(x))
    
    def decode(self, x: mx.array) -> mx.array:
        return self.dec(x)
        