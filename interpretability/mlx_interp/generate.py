# Copyright © 2025 Apple Inc.

import argparse
import json
import sys

import mlx.core as mx

from mlx_lm import load

from .activation_cache import ActivationCache
from .patching import PatchedLayer, patch_module

DEFAULT_PROMPT = "hello"
DEFAULT_MODEL = "mlx-community/gemma-2-2b"


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--activations-path",
        type=str,
        help="Path to activations to cache, in dot notation with wildcards."
    )
    parser.add_argument(
        "--activations-output-file",
        type=str,
        help="File to store activations at (in safetensors format)"
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--chat-template-config",
        help="Additional config for `apply_chat_template`. Should be a dictionary of"
        " string keys to values represented as a JSON decodable string.",
        default=None,
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    tokenizer_config = {}
    tokenizer_config["trust_remote_code"] = True

    model_path = args.model
    model_path = model_path or DEFAULT_MODEL

    model, tokenizer = load(
        model_path,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
    )

    template_kwargs = {}
    if args.chat_template_config is not None:
        template_kwargs = json.loads(args.chat_template_config)

    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    prompt = sys.stdin.read() if prompt == "-" else prompt
    if not args.ignore_chat_template and tokenizer.chat_template is not None:
        if args.system_prompt is not None:
            messages = [{"role": "system", "content": args.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            **template_kwargs,
        )

        tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        tokenized_prompt = tokenizer.encode(prompt)

    cache = ActivationCache(model_name=model_path, prompt=prompt)
    patch_module(model, args.activations_path, PatchedLayer, cache)
    model(mx.array(tokenized_prompt)[None])
    cache.remove_batch_dim()
    cache.save_safetensors(args.activations_output_file)
    

if __name__ == "__main__":
    main()
