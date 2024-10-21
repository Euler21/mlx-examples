import mlx.nn as nn
import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizer
import plotly.graph_objects as go

import numpy as np

from typing import List, Union

class PatchedLayer(nn.Module):
    def __init__(self, layer: nn.Module):
        if isinstance(layer, PatchedLayer):
            # Avoid double patching
            layer = layer["layer"]
        self["layer"] = layer
        object.__setattr__(self, "filter_and_map", layer.filter_and_map)
        object.__setattr__(self, "activation_cache", None)
    def __repr__(self):
        return f"({type(self).__name__}) {repr(self["layer"])}"
    def __call__(self, *args, **kwargs):
        out = self["layer"](*args, **kwargs)
        self.cache(out)
        return out
    def __getattr__(self, attr):
        return getattr(self["layer"], attr)
    def __setattr__(self, attr, value):
        setattr(self["layer"], attr, value)
    def __delattr__(self, attr):
        delattr(self["layer"], attr)
    def cache(self, activations):
        object.__setattr__(self, "activation_cache", activations)

def patch_dict(m: dict | nn.Module):
    if isinstance(m, nn.Module):
        return PatchedLayer(m)
    if isinstance(m, dict):
        for k, v in m.items():
            m[k] = patch_dict(v)
    if isinstance(m, list):
        for i, v in enumerate(m):
            m[i] = patch_dict(v)
    return m

def patch_layers(model: nn.Module, layers: List[int] = None):
    if layers == None:
        layers = range(len(model.layers))
    for i in layers:
        model.layers[i] = PatchedLayer(model.layers[i])

def layer_lens(model: nn.Module, 
               tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
               prompt: str, 
               lens: nn.Module):
    patch_layers(model)
    inputs = mx.array(tokenizer.encode(prompt))
    out = model(inputs[None])
    for i, layer in enumerate(model.layers):
        print(f"Layer {i} prediction")

def logit_lens(model: nn.Module, 
               tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper], 
               prompt: str):
    patch_layers(model)
    inputs = mx.array(tokenizer.encode(prompt))
    # Throw away batch dim
    out = model(inputs[None])[0] # (seq_len, h)
    probs = []
    preds = []
    for i, layer in enumerate(model.layers):
        layer_out = model.model.norm(layer.activation_cache)[0] # (seq_len, h)
        layer_out = model.model.embed_tokens.as_linear(layer_out) # (seq_len, vocab_size)
        layer_out = mx.tanh(layer_out / model.final_logit_softcapping)
        layer_out = layer_out * model.final_logit_softcapping
        layer_out = layer_out - mx.logsumexp(layer_out, axis=1, keepdims=True)
        probs.append(layer_out.max(axis=1).tolist())
        preds.append(layer_out.argmax(axis=1).tolist())
        #print(f"Layer {i} prediction ({layer_pred.max(axis=1).tolist():.4f}): {tokenizer.decode([layer_pred.argmax(axis=1).tolist()])}")
    # pred = out[0] # (seq_len, vocab_size)
    # pred = pred - mx.logsumexp(pred, axis=1)
    #print(f"Final prediction ({pred.max(axis=1).tolist():.4f}): {tokenizer.decode([pred.argmax(axis=1).tolist()])}")
    create_lens_heatmap(np.array(preds), np.array(probs), "Probability").show()

def create_lens_heatmap(preds, stats, stat_name):
    """
    Create a logit lens heatmap visualization.
    
    :param predictions: 2D array of shape (num_layers, sequence_length) containing predicted token strings
    :param logit_scores: 2D array of shape (num_layers, sequence_length) containing logit scores
    :return: Plotly Figure object
    """
    num_layers, seq_length = preds.shape
    
    # Create text annotations for the heatmap
    text = [[f"{preds[i, j]}<br>{stat_name}: {stats[i, j]:.2f}"
             for j in range(seq_length)] for i in range(num_layers)]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=stats,
        text=text,
        hoverinfo='text',
        colorscale='Viridis',
        colorbar=dict(title='Logit Score')
    ))
    
    # Update layout
    fig.update_layout(
        title='Logit Lens Visualization',
        xaxis=dict(title='Sequence Position'),
        yaxis=dict(title='Transformer Layer', autorange='reversed'),
    )
    
    return fig