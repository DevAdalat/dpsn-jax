import jax
import jax.numpy as jnp
from flax import linen as nn
import math

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:

    def flash_attention(*args, **kwargs):
        raise ImportError("Flash Attention not available")

    FLASH_ATTENTION_AVAILABLE = False


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

        # Initial Reshape: (Batch, SeqLen, Heads, HeadDim)
        q = q.reshape(batch, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, head_dim)

        use_flash = FLASH_ATTENTION_AVAILABLE
        try:
            if use_flash and jax.devices()[0].platform != "tpu":
                use_flash = False
        except:
            pass

        sm_scale = 1.0 / math.sqrt(head_dim)

        if use_flash:
            # Pallas Flash Attention expects (Batch, Heads, SeqLen, HeadDim)
            # Transpose to match expectation
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            # Output is (Batch, Heads, SeqLen, HeadDim)
            attn_output = flash_attention(q, k, v, causal=True, sm_scale=sm_scale)

            # Transpose back to (Batch, SeqLen, Heads, HeadDim)
            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        else:
            # CPU Fallback also expects (Batch, Heads, SeqLen, HeadDim)
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * sm_scale

            if mask is not None:
                big_neg = -1e9
                logits = jnp.where(mask, logits, big_neg)

            weights = nn.softmax(logits, axis=-1)
            attn_output = jnp.einsum("bhqk,bhkd->bhqd", weights, v)

            # Transpose back to (Batch, SeqLen, Heads, HeadDim)
            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

        attn_output = attn_output.reshape(batch, seq_len, self.d_model)

        output = nn.Dense(self.d_model, dtype=self.dtype)(attn_output)

        return output
