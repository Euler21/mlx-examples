import mlx.nn as nn
import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizer
from .patching import PatchedLayer, patch_layers
from mlx_lm import load, generate

import plotly.graph_objects as go

import numpy as np

from typing import List, Union


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
    inputs = mx.array(tokenizer.encode(prompt))[None]
    input_tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())
    
    # Throw away batch dim
    out = model(inputs)[0] # (seq_len, h)
    probs = []
    preds = []
    for i, layer in enumerate(model.layers):
        layer_out = model.model.norm(layer.activation_cache)[0] # (seq_len, h)
        layer_out = model.model.embed_tokens.as_linear(layer_out) # (seq_len, vocab_size)
        layer_out = mx.tanh(layer_out / model.final_logit_softcapping)
        layer_out = layer_out * model.final_logit_softcapping
        layer_out = layer_out - mx.logsumexp(layer_out, axis=1, keepdims=True)
        layer_out = mx.exp(layer_out)
        probs.append(layer_out.max(axis=1).tolist())
        preds.append(tokenizer.convert_ids_to_tokens(layer_out.argmax(axis=1).tolist()))
    create_lens_heatmap(np.array(input_tokens), np.array(preds), np.array(probs), "Probability").show()

def create_lens_heatmap(input_seq, preds, stats, stat_name):
    """
    Create a logit lens heatmap visualization.
    
    :param predictions: 2D array of shape (num_layers, sequence_length) containing predicted token strings
    :param logit_scores: 2D array of shape (num_layers, sequence_length) containing logit scores
    :return: Plotly Figure object
    """
    num_layers, seq_length = preds.shape
    z_data = stats
    text_data = preds
    # Hack to ensure that Plotly doesn't de-duplicate the x-axis labels
    x_labels = [x + "\u200c" * i for i, x in enumerate(input_seq)]
    x_top_labels = x_labels[1:]
    hover_text = [[f"Layer: {i}<br>Token: {preds[i, j]}<br>{stat_name}: {stats[i, j]:.2f}"
                   for j in range(seq_length)] for i in range(0, num_layers)]
    

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        text=text_data,
        hovertext=hover_text,
        texttemplate="%{text}",
        hoverinfo='text',
        colorscale='Viridis',
        colorbar=dict(title='Logit Score'),
        textfont=dict(size=12)
    ))

    # Add annotations for top labels
    for i, label in enumerate(x_top_labels):
        fig.add_annotation(
            x=i,
            y=1.05,  # Position above the heatmap; adjust as needed
            xref='x',
            yref='paper',
            text=label,
            showarrow=False,
            font=dict(color='black'),
            xanchor='center',
            yanchor='top'
        )
    
    # Update layout
    fig.update_layout(
        title='Logit Lens Visualization',
        margin=dict(t=80, l=80, r=0, b=80),
        xaxis=dict(title='Sequence Position'),
        yaxis=dict(title='Transformer Layer'),
        height=max(500, num_layers * 30),  # Adjust height based on number of layers
        width=max(800, seq_length * 100),   # Adjust width based on sequence length
    )
    
    return fig


def main():
    model, tokenizer = load("mlx-community/gemma-2-2b-it")
    prompt = "What is written on the Ring?"
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "One Ring to rule them all, One Ring to find them, One Ring to bring them all and in the darkness bind"
    logit_lens(model, tokenizer, prompt)

if __name__ == "__main__":
    main()