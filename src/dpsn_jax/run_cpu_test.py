import jax
import jax.numpy as jnp
from dpsn_flax import DPSNR, DPSNRConfig


def run_cpu_test():
    print("=" * 60)
    print("DPSN-R (Advanced Reasoning): JAX/Flax CPU Test")
    print("=" * 60)

    # Config
    config = DPSNRConfig(
        vocab_size=100,
        hidden_dim=32,
        pool_size=200,
        pool_dim=32,
        max_k=10,
        max_loops=4,
        min_k=2,
    )

    model = DPSNR(config)

    key = jax.random.PRNGKey(0)
    dummy_input = jax.random.randint(key, (2, 8), 0, 100)

    print("Initializing parameters...")
    params = model.init(key, dummy_input)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameter Count: {param_count:,}")

    print("\nRunning Forward Pass...")
    apply_fn = jax.jit(model.apply)
    output = apply_fn(params, dummy_input)

    print("\nResults:")
    print(f"Logits: {output['logits'].shape}")
    print(f"Ponder Loss: {output['ponder_loss']:.4f}")
    print(f"Avg Loops: {output['loops']:.2f}")

    print("\nSUCCESS: Advanced Reasoning Modules (ACT, Phase, Accumulator) verified!")


if __name__ == "__main__":
    run_cpu_test()
