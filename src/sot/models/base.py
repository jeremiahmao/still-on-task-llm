"""Qwen2.5-3B model loading and tokenizer setup."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    name: str = "Qwen/Qwen2.5-3B",
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer.

    Returns the model in eval mode by default.
    """
    torch_dtype = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    # Qwen tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batch generation

    model.eval()
    return model, tokenizer
