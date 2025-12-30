import jax
import jax.numpy as jnp
from flax import linen as nn
import math

# Default fallback
FLASH_ATTENTION_AVAILABLE = False
flash_attention = None

try:
    # Attempt import
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as fa_fn

    flash_attention = fa_fn
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    pass


class FlashAttention(nn.Module):
    num_heads: int
    d_model: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, training: bool = False):
        batch, seq_len, _ = x.shape
        head_dim = self.d_model // self.num_heads

        q = nn.Dense(self.d_model, dtype=self.dtype)(x)
        k = nn.Dense(self.d_model, dtype=self.dtype)(x)
        v = nn.Dense(self.d_model, dtype=self.dtype)(x)

        q = q.reshape(batch, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, head_dim)

        use_flash = FLASH_ATTENTION_AVAILABLE
        try:
            if use_flash and jax.devices()[0].platform != "tpu":
                use_flash = False
        except:
            pass

        # Pallas Flash Attention requires block size constraints.
        # Default MIN_BLOCK_SIZE is often 128. If seq_len < 128, it fails.
        # For simplicity and stability, we fallback to standard attention for short sequences.
        if use_flash and seq_len < 128:
            use_flash = False

        sm_scale = 1.0 / math.sqrt(head_dim)

        if use_flash and flash_attention is not None:
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            attn_output = flash_attention(q, k, v, causal=True, sm_scale=sm_scale)

            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        else:
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * sm_scale

            if mask is not None:
                big_neg = -1e9
                logits = jnp.where(mask, logits, big_neg)

            weights = nn.softmax(logits, axis=-1)
            attn_output = jnp.einsum("bhqk,bhkd->bhqd", weights, v)

            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

        attn_output = attn_output.reshape(batch, seq_len, self.d_model)

        output = nn.Dense(self.d_model, dtype=self.dtype)(attn_output)

        return output
