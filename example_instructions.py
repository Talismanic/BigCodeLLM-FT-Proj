# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
from typing import List, Tuple

import fire
import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils import InfiniteDataLoader, chunk


# Set relative paths for loading model and tokenizer
MODEL_DIR = "./CodeLlama-7b-Python"
TOKENIZER_DIR = "./CodeLlama-7b-Python"


def get_prompt_meta_infilling(prefix, suffix, tokenizer):
    prompt = f"<PRE>{prefix}<SUF>{suffix}<MID>"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    inputs = torch.tensor([prompt_tokens])  # Add batch dimension
    return prompt, inputs


def get_prompt_infilling(prefix, suffix, tokenizer):
    """
    Generates infilling prompt format for Code Llama
    """
    prompt = f"<PRE>{prefix}<SUF>{suffix}<MID>"
    return prompt


def get_prompt_generation(prompt, tokenizer):
    """
    Generates plain generation prompt format for Code Llama
    """
    return prompt


def code_generate(prompt, model, tokenizer, max_tokens):
    """
    Generate code completion from a simple prompt
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original prompt from the generated text
    generated_code = generated_text.replace(prompt, "")
    return generated_code


def code_infilling(prefix, suffix, model, tokenizer, max_tokens):
    """
    Generate code infilling given prefix and suffix
    """
    prompt = f"<PRE>{prefix}<SUF>{suffix}<MID>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The generated infilled code should start after <MID>
    infilled_code = generated_text.replace(prompt, "")
    
    return infilled_code


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: int = 128,
    max_batch_size: int = 4,
):
    """
    Runs inference on the Model with default parameters
    
    Args:
        ckpt_dir: Path to the model checkpoint directory
        tokenizer_path: Path to the tokenizer directory
        temperature: Temperature for sampling (0.2 is useful for code)
        top_p: Top-p for sampling
        max_seq_len: Maximum sequence length for input context
        max_gen_len: Maximum generation length
        max_batch_size: Maximum batch size for generation
    """

    # Load model and tokenizer
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    
    # Example prompts for code generation
    generation_prompts = [
        "# Python function to calculate fibonacci numbers:\ndef fibonacci(n):",
        "# Quick sort algorithm in Python:\ndef quick_sort(arr):",
        "# Create a REST API using FastAPI\nfrom fastapi import FastAPI\n",
    ]
    
    print("\n=== Code Generation Examples ===")
    for prompt in generation_prompts:
        print(f"\nPrompt: {prompt}")
        result = code_generate(prompt, model, tokenizer, max_gen_len)
        print(f"Generated:\n{result}")
    
    # Example prompts for code infilling
    infilling_examples = [
        ("def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    ", 
         "\n    return left + middle + right"),
        ("class Node:\n    def __init__(self, data):\n        ",
         "\n    def __repr__(self):\n        return str(self.data)"),
    ]
    
    print("\n=== Code Infilling Examples ===")
    for prefix, suffix in infilling_examples:
        print(f"\nPrefix: {prefix}")
        print(f"Suffix: {suffix}")
        infilled = code_infilling(prefix, suffix, model, tokenizer, max_gen_len)
        print(f"Infilled: {infilled}")


if __name__ == "__main__":
    fire.Fire(main)