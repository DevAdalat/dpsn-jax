import jax
import jax.numpy as jnp
import os
import json
import pickle
import time
from typing import Dict, Any

from .model import DPSNModel
from .config import Config


def load_vocab(vocab_path: str) -> Dict:
    with open(vocab_path, "r") as f:
        data = json.load(f)
    return data


def generate(config_path: str, checkpoint_path: str, prompt: str, length: int = 100):
    from .config import load_config

    config = load_config(config_path)

    vocab_data = load_vocab(config.data.vocab_path)
    stoi = vocab_data["stoi"]
    itos = {int(k): v for k, v in vocab_data["itos"].items()}
    vocab_size = len(stoi)

    model = DPSNModel(
        vocab_size=vocab_size,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_memory_slots=config.model.num_memory_slots,
        min_k=config.model.min_k,
        max_k=config.model.max_k,
    )

    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    params = data["params"]

    input_indices = [stoi.get(c, 0) for c in prompt]
    context_window = config.training.seq_len

    print(f"Generating from prompt: '{prompt}'")
    generated = list(input_indices)

    rng = jax.random.PRNGKey(0)

    # ---------------------------------------------------------
    # JIT-compiled inference step for speed
    # ---------------------------------------------------------
    @jax.jit
    def inference_step(params, input_seq, rng):
        # input_seq: (1, seq_len)
        logits = model.apply(
            {"params": params}, input_seq, training=False, rngs={"noise": rng}
        )
        # logits: (1, seq_len, vocab_size)
        # We only care about the last token's logits
        next_token_logits = logits[0][-1]
        return next_token_logits

    # Warmup compilation
    print("Compiling inference step...")
    dummy_input = jnp.zeros((1, context_window), dtype=jnp.int32)
    _ = inference_step(params, dummy_input, rng)
    print("Compilation complete.")

    start_time = time.time()

    for i in range(length):
        # Prepare input (pad or truncate to exact context_window)
        curr_seq = generated[-context_window:]
        if len(curr_seq) < context_window:
            # Pad with 0s (assuming 0 is safe/padding, though typically we'd care more)
            # For casual generation, just left-pad
            curr_seq = [0] * (context_window - len(curr_seq)) + curr_seq

        curr_input_tensor = jnp.array([curr_seq], dtype=jnp.int32)

        rng, step_rng = jax.random.split(rng)

        # Fast inference
        next_token_logits = inference_step(params, curr_input_tensor, step_rng)

        # Greedy decode
        next_token = jnp.argmax(next_token_logits)
        token_int = int(next_token)

        generated.append(token_int)
        print(itos.get(token_int, ""), end="", flush=True)

    total_time = time.time() - start_time
    tokens_per_sec = length / total_time
    print(f"\n\nGeneration complete. Speed: {tokens_per_sec:.2f} tokens/sec")

    text = "".join([itos.get(i, "") for i in generated])
    return text
