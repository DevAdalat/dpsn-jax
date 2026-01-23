import pytest
import numpy as np
import jax
import jax.numpy as jnp
from dpsn_jax.dpsn_flax import DPSNRConfig
from dpsn_jax.train_jax import train_loop


def test_loss_decreases():
    """Verify model learns by checking loss decreases over 100 steps."""
    config = DPSNRConfig.nano()
    # Reduce size further for speed
    config.max_loops = 2
    config.max_seq_len = 16
    config.min_k = 2
    config.max_k = 4

    # Override data generator with learnable pattern (Repeat Task)
    # [A, B, C, A, B, C] -> Predict next token is easy
    def create_structured_data(key, batch_size, seq_len, vocab_size):
        # Generate random start, then repeat
        half_len = seq_len // 2
        data = jax.random.randint(key, (batch_size, half_len), 0, vocab_size)
        return jnp.concatenate([data, data], axis=1)

    # Monkey patch the module's function
    import dpsn_jax.train_jax

    original_fn = dpsn_jax.train_jax.create_synthetic_data
    dpsn_jax.train_jax.create_synthetic_data = create_structured_data

    try:
        # Train for 100 steps (increased from 50)
        history = train_loop(
            config, num_steps=100, batch_size=4
        )  # increased batch size
    finally:
        # Restore original function
        dpsn_jax.train_jax.create_synthetic_data = original_fn

    losses = [m["loss"] for m in history]

    # Check if final loss is lower than initial loss (smoothed)
    # Use median to be robust to spikes
    initial_loss = np.median(losses[:10])
    final_loss = np.median(losses[-10:])

    assert final_loss < initial_loss, (
        f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )

    # Check if metrics contain expected keys
    assert "ponder_loss" in history[0]
    assert "ce_loss" in history[0]
    assert "loops" in history[0]
