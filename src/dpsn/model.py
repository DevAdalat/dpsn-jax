import jax
import jax.numpy as jnp
from flax import linen as nn
from .dpsn_layer import DPSNBlock


class DPSNModel(nn.Module):
    vocab_size: int
    d_model: int
    num_layers: int
    num_memory_slots: int
    min_k: int
    max_k: int
    router_dim: int = 0  # Add router_dim support

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        # x: (batch, seq_len) - integer indices

        # Embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)

        # Layers
        total_aux_loss = 0.0
        total_active_count = 0.0
        for _ in range(self.num_layers):
            x, (aux_loss, active_count) = DPSNBlock(
                d_model=self.d_model,
                num_memory_slots=self.num_memory_slots,
                min_k=self.min_k,
                max_k=self.max_k,
                router_dim=self.router_dim,  # Pass it down
            )(x, training=training)
            total_aux_loss += aux_loss
            total_active_count += active_count

        # Final Norm
        x = nn.LayerNorm()(x)

        # Output Head
        logits = nn.Dense(features=self.vocab_size)(x)

        return logits, (total_aux_loss, total_active_count)
