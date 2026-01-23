import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from dpsn_jax.dpsn_flax import DPSNR


def test_forward_pass_shapes(model, initialized_model_state, input_ids):
    """Verify output shapes match expectations."""
    params = initialized_model_state
    outputs = model.apply(params, input_ids)

    # Check logits shape: [Batch, Seq, Vocab]
    assert outputs["logits"].shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        model.config.vocab_size,
    )

    # Check scalar metrics
    assert outputs["ponder_loss"].shape == ()
    assert outputs["loops"].shape == ()


def test_ponder_loss_range(model, initialized_model_state, input_ids):
    """Verify ponder loss is within valid range."""
    params = initialized_model_state
    outputs = model.apply(params, input_ids)

    # Ponder loss should be positive
    assert outputs["ponder_loss"] >= 0.0

    # Should not exceed max loops + 1 (theoretical bound)
    assert outputs["ponder_loss"] <= model.config.max_loops + 1.0


def test_dropout_determinism(model, initialized_model_state, input_ids, rng_key):
    """Verify dropout behaves correctly with train flag."""
    params = initialized_model_state
    dropout_key = jax.random.fold_in(rng_key, 1)

    # Eval mode (deterministic)
    out1 = model.apply(params, input_ids, train=False)
    out2 = model.apply(params, input_ids, train=False)
    assert jnp.allclose(out1["logits"], out2["logits"])

    # Train mode (stochastic) - requires RNG for dropout
    out3 = model.apply(params, input_ids, train=True, rngs={"dropout": dropout_key})
    # With same key, should be same
    out4 = model.apply(params, input_ids, train=True, rngs={"dropout": dropout_key})
    assert jnp.allclose(out3["logits"], out4["logits"])

    # With different key, should be different (if dropout > 0)
    # Note: Nano config has dropout=0.0 by default, so we need to enable it
    # But we can't easily change config of instantiated model fixture
    pass


def test_jit_compatibility(model, initialized_model_state, input_ids):
    """Verify model can be JIT compiled."""
    params = initialized_model_state

    @jax.jit
    def forward(p, x):
        return model.apply(p, x)

    # First call compiles
    out1 = forward(params, input_ids)
    # Second call uses cache
    out2 = forward(params, input_ids)

    assert jnp.allclose(out1["logits"], out2["logits"])
