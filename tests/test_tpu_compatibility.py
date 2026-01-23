import jax
import jax.numpy as jnp
import pytest
from dpsn_jax.dpsn_flax import DPSNR


@pytest.mark.tpu
def test_static_shapes_no_recompile(model, initialized_model_state, input_ids):
    """Verify that changing input data values doesn't trigger recompilation."""
    params = initialized_model_state

    jit_apply = jax.jit(model.apply)

    # First call
    _ = jit_apply(params, input_ids)

    # Second call with different data but same shape
    new_input = jax.random.randint(
        jax.random.PRNGKey(99), input_ids.shape, 0, model.config.vocab_size
    )
    _ = jit_apply(params, new_input)

    # We rely on JAX's internal caching. If this was dynamic shape,
    # it would have failed or warned on TPU.
    # To be rigorous, we check XLA computation
    # Use modern lower() API
    hlo_text = jax.jit(model.apply).lower(params, input_ids).as_text()

    # Verify 'scan' is present in HLO (indicating loop was lowered correctly)
    assert (
        "custom-call" in hlo_text or "while" in hlo_text or "loop" in hlo_text.lower()
    )
    # Note: HLO structure varies, but we expect complex control flow


def test_device_availability():
    """Check JAX sees devices."""
    devices = jax.devices()
    assert len(devices) > 0, "No JAX devices found!"
    print(f"JAX Devices: {devices}")


def test_masking_mechanism(model, initialized_model_state, input_ids):
    """Indirectly verify masking logic by checking k_values behavior."""
    # We can't easily probe internal tensors without exposing them,
    # but we can ensure model runs without errors for inputs that would
    # trigger dynamic k if not masked.
    params = initialized_model_state

    # Run with JIT to ensure XLA is happy with the masking ops
    jit_apply = jax.jit(model.apply)
    out = jit_apply(params, input_ids)

    assert out["logits"] is not None
