import jax
import jax.numpy as jnp
from dpsn.model import DPSNModel
from flax.traverse_util import flatten_dict


def inspect_params():
    model = DPSNModel(
        vocab_size=100,
        d_model=32,
        num_layers=1,
        num_memory_slots=100,
        min_k=5,
        max_k=10,
    )
    init_rngs = {"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(1)}
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    variables = model.init(init_rngs, dummy_input, training=False)
    params = variables["params"]

    flat_params = flatten_dict(params)
    for k, v in flat_params.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")


if __name__ == "__main__":
    inspect_params()
