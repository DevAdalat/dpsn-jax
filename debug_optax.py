import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


def test_optax():
    params = {"a": jnp.array([1.0, 2.0])}
    tx = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=lambda x, y: x, params=params, tx=tx)

    grads = {"a": jnp.array([0.1, 0.1])}
    state = state.apply_gradients(grads=grads)
    print("Optax step success")


if __name__ == "__main__":
    test_optax()
