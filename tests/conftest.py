import pytest
import jax
import jax.numpy as jnp
from dpsn_jax.dpsn_flax import DPSNR, DPSNRConfig


@pytest.fixture
def nano_config():
    """Returns a nano configuration for testing."""
    return DPSNRConfig.nano()


@pytest.fixture
def model(nano_config):
    """Returns an initialized DPSNR model."""
    return DPSNR(nano_config)


@pytest.fixture
def rng_key():
    """Returns a JAX RNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def input_ids(nano_config, rng_key):
    """Returns a batch of random input IDs."""
    batch_size = 2
    seq_len = 32
    return jax.random.randint(rng_key, (batch_size, seq_len), 0, nano_config.vocab_size)


@pytest.fixture
def initialized_model_state(model, input_ids, rng_key):
    """Returns initialized variables (params)."""
    return model.init(rng_key, input_ids)
