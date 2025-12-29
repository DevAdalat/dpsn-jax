import jax
import jax.numpy as jnp
from flax import linen as nn
from .router import Router


class DPSNLayer(nn.Module):
    d_model: int
    num_memory_slots: int
    min_k: int
    max_k: int
    router_dim: int = 0  # Add router_dim with default 0

    def setup(self):
        self.param_pool = self.param(
            "param_pool",
            nn.initializers.normal(stddev=0.02),
            (self.num_memory_slots, self.d_model),
        )

        self.router = Router(
            num_memory_slots=self.num_memory_slots,
            min_k=self.min_k,
            max_k=self.max_k,
            router_dim=self.router_dim,  # Pass it to Router
        )

    def __call__(self, x: jnp.ndarray, training: bool = False):
        indices, weights, _, aux_loss = self.router(x, training=training)
        selected_params = self.param_pool[indices]
        proj = jnp.einsum("...d,...kd->...k", x, selected_params)
        weighted_proj = proj * weights
        output = jnp.einsum("...k,...kd->...d", weighted_proj, selected_params)

        # Count unique parameters updated (selected) in this batch
        # indices shape: (batch, seq, max_k)
        flat_indices = indices.reshape(-1)
        # Use a boolean mask to count unique slots
        touched_mask = jnp.zeros((self.num_memory_slots,), dtype=jnp.bool_)
        touched_mask = touched_mask.at[flat_indices].set(True)
        active_count = jnp.sum(touched_mask)

        return output, (aux_loss, active_count)


class DPSNBlock(nn.Module):
    d_model: int
    num_memory_slots: int
    min_k: int
    max_k: int
    router_dim: int = 0  # Add router_dim to block

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        norm_x = nn.LayerNorm()(x)

        mask = nn.make_causal_mask(
            jnp.ones(x.shape[:2], dtype=jnp.int32), dtype=jnp.bool_
        )

        attn_out = nn.SelfAttention(num_heads=8)(norm_x, mask=mask)
        x = x + attn_out

        norm_x = nn.LayerNorm()(x)
        dpsn_out, metrics = DPSNLayer(
            d_model=self.d_model,
            num_memory_slots=self.num_memory_slots,
            min_k=self.min_k,
            max_k=self.max_k,
            router_dim=self.router_dim,  # Pass it down
        )(norm_x, training=training)

        return x + dpsn_out, metrics
