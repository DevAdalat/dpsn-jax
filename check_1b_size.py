import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from dpsn.model import DPSNModel
from dpsn.config import Config
import yaml


def calculate_model_size_and_memory():
    # Load 1B config
    with open("configs/config_1b.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    # Manually reconstruction config object for size calculation
    # Assuming vocab size approx 32k or 50k (standard)
    vocab_size = 32000

    d_model = config_dict["model"]["d_model"]
    num_layers = config_dict["model"]["num_layers"]
    num_memory_slots = config_dict["model"]["num_memory_slots"]
    router_dim = config_dict["model"]["router_dim"]

    # 1. Embeddings
    embedding_params = vocab_size * d_model

    # 2. Per Layer Stats
    # Attention: 4 * d_model^2 (Q, K, V, Out)
    attn_params = 4 * (d_model * d_model)

    # Layer Norms (2 per block): 2 * d_model * 2 (scale + bias)
    norm_params = 4 * d_model

    # Router:
    # Complexity Net: d_model * 1 + 1
    # Hidden: d_model * router_dim + router_dim
    # Scores: router_dim * num_memory_slots + num_memory_slots
    router_params = (
        (d_model * 1 + 1)
        + (d_model * router_dim + router_dim)
        + (router_dim * num_memory_slots + num_memory_slots)
    )

    # Parameter Pool (The Big One): num_memory_slots * d_model
    pool_params = num_memory_slots * d_model

    # Total per layer
    layer_params = attn_params + norm_params + router_params + pool_params

    # 3. Output Head
    output_params = d_model * vocab_size

    # Total Model
    total_params = embedding_params + (num_layers * layer_params) + output_params

    # Memory Estimation (Float32 = 4 bytes)
    size_gb = (total_params * 4) / (1024**3)

    # Optimizer Overhead (Adam = 2 states + 1 param copy = 12 bytes/param usually,
    # but strictly for storage: Param + Moment1 + Moment2 = 3x)
    training_mem_gb = size_gb * 3

    print("=" * 60)
    print(f"DPSN 1B Configuration Analysis")
    print("=" * 60)
    print(f"D_Model: {d_model}")
    print(f"Layers: {num_layers}")
    print(f"Memory Slots: {num_memory_slots}")
    print(f"Router Dim: {router_dim}")
    print("-" * 60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size (FP32): {size_gb:.2f} GB")
    print(f"Est. Training Memory (Params + Opt States): {training_mem_gb:.2f} GB")
    print("=" * 60)

    if training_mem_gb < 16.0:
        print("✅ SUCCESS: This configuration fits comfortably in 16GB RAM.")
    elif size_gb < 16.0:
        print("⚠️ WARNING: Model fits, but training might OOM with Adam states.")
        print("   Consider using Adafactor or mixed precision.")
    else:
        print("❌ FAILURE: Model is too large for 16GB RAM.")


if __name__ == "__main__":
    calculate_model_size_and_memory()
