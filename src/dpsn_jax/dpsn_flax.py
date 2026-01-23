import dataclasses
from enum import IntEnum
from typing import Any, Callable, Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
import math


class ReasoningPhase(IntEnum):
    UNDERSTANDING = 0
    REASONING = 1
    EXPRESSION = 2


@dataclasses.dataclass
class DPSNRConfig:
    vocab_size: int = 1000
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    controller_ff_multiplier: float = 2.0
    pool_size: int = 1000
    pool_dim: int = 64
    max_loops: int = 4
    max_seq_len: int = 128

    # Retrieval Hyperparams
    min_k: int = 2
    max_k: int = 10

    dropout_rate: float = 0.0
    halt_threshold: float = 0.99
    ponder_loss_weight: float = 0.01

    @property
    def total_params(self) -> int:
        """Estimate total parameters."""
        # Controller
        embed = self.vocab_size * self.hidden_dim
        # Attention: 4 * d_model^2 * layers
        attn = 4 * (self.hidden_dim**2) * self.num_layers
        # FFN: 2 * d_model * d_ff * layers
        d_ff = int(self.hidden_dim * self.controller_ff_multiplier)
        ffn = 2 * self.hidden_dim * d_ff * self.num_layers

        controller = embed + attn + ffn

        # Pool
        pool = self.pool_size * self.pool_dim

        # Components
        # Halt: hidden -> hidden//4 -> 1
        halt = (self.hidden_dim * (self.hidden_dim // 4)) + (self.hidden_dim // 4)
        # Phase: hidden -> hidden//4 -> 3
        phase = (self.hidden_dim * (self.hidden_dim // 4)) + (
            (self.hidden_dim // 4) * 3
        )
        # Accumulator: 2*hidden -> hidden (gate), hidden -> hidden (transform)
        accum = (2 * self.hidden_dim * self.hidden_dim) + (
            self.hidden_dim * self.hidden_dim
        )
        # Integrator: 2*hidden -> hidden
        integ = 2 * self.hidden_dim * self.hidden_dim
        # Loop Embed: max_loops * hidden
        loop = self.max_loops * self.hidden_dim

        return controller + pool + halt + phase + accum + integ + loop

    @property
    def active_params_per_token(self) -> int:
        """Estimate active parameters per token."""
        # Controller is always active
        controller = self.total_params - (self.pool_size * self.pool_dim)

        # Average k retrieved vectors
        avg_k = (self.min_k + self.max_k) // 2
        pool_active = avg_k * self.pool_dim

        # Multiplied by average loops (assume half of max)
        avg_loops = max(1, self.max_loops // 2)

        return controller + (pool_active * avg_loops)

    @classmethod
    def nano(cls) -> "DPSNRConfig":
        return cls(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            pool_size=1000,
            pool_dim=64,
            max_loops=4,
            max_seq_len=128,
            min_k=4,
            max_k=16,
        )

    @classmethod
    def micro(cls) -> "DPSNRConfig":
        return cls(
            vocab_size=50257,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            pool_size=10_000,
            pool_dim=128,
            max_loops=4,
            max_seq_len=256,
            min_k=8,
            max_k=32,
        )

    @classmethod
    def mini(cls) -> "DPSNRConfig":
        return cls(
            vocab_size=50257,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            pool_size=100_000,
            pool_dim=256,
            max_loops=8,
            max_seq_len=512,
            min_k=16,
            max_k=64,
        )

    @classmethod
    def small(cls) -> "DPSNRConfig":
        """Approx 400M params."""
        return cls(
            vocab_size=50257,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            pool_size=400_000,
            pool_dim=768,
            max_loops=12,
            max_seq_len=1024,
            min_k=32,
            max_k=128,
        )

    @classmethod
    def medium(cls) -> "DPSNRConfig":
        """Approx 600M params."""
        return cls(
            vocab_size=50257,
            hidden_dim=1024,
            num_layers=16,
            num_heads=16,
            pool_size=400_000,
            pool_dim=1024,
            max_loops=16,
            max_seq_len=1024,
            min_k=32,
            max_k=128,
        )

    @classmethod
    def large(cls) -> "DPSNRConfig":
        """Approx 1B params."""
        return cls(
            vocab_size=50257,
            hidden_dim=1280,
            num_layers=24,
            num_heads=20,
            pool_size=500_000,
            pool_dim=1280,
            max_loops=16,
            max_seq_len=2048,
            min_k=64,
            max_k=256,
        )


# --- Components ---


class TransformerBlock(nn.Module):
    config: DPSNRConfig

    def setup(self):
        self.layernorm1 = nn.LayerNorm()
        self.self_attn = nn.SelfAttention(
            num_heads=self.config.num_heads,
            decode=False,
        )
        self.dropout1 = nn.Dropout(rate=self.config.dropout_rate)
        self.layernorm2 = nn.LayerNorm()
        self.dense1 = nn.Dense(
            int(self.config.hidden_dim * self.config.controller_ff_multiplier)
        )
        self.dense2 = nn.Dense(self.config.hidden_dim)
        self.dropout2 = nn.Dropout(rate=self.config.dropout_rate)

    def __call__(self, x, train: bool = False):
        batch, seq, _ = x.shape
        mask = nn.make_causal_mask(jnp.ones((batch, seq), dtype=jnp.int32))

        y = self.layernorm1(x)
        y = self.self_attn(y, mask=mask)
        y = self.dropout1(y, deterministic=not train)
        x = x + y

        y = self.layernorm2(x)
        y = self.dense1(y)
        y = nn.gelu(y)
        y = self.dense2(y)
        y = self.dropout2(y, deterministic=not train)
        x = x + y
        return x


class TinyController(nn.Module):
    config: DPSNRConfig

    def setup(self):
        self.layers = [
            TransformerBlock(self.config) for _ in range(self.config.num_layers)
        ]

    def __call__(self, x, train: bool = False):
        for layer in self.layers:
            x = layer(x, train=train)
        return x


class HaltPredictor(nn.Module):
    hidden_dim: int
    halt_threshold: float = 0.99

    @nn.compact
    def __call__(self, hidden):
        # hidden: [B, T, D]
        x = nn.Dense(self.hidden_dim // 4)(hidden)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return nn.sigmoid(x)  # [B, T, 1]


class PhaseClassifier(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, hidden):
        # hidden: [B, T, D]
        # Pool over sequence for global phase (as per original design)
        pooled = hidden.mean(axis=1)  # [B, D]

        x = nn.Dense(self.hidden_dim // 4)(pooled)
        x = nn.gelu(x)
        logits = nn.Dense(3)(x)  # [B, 3] (3 phases)

        # Determine global phase (Mode of batch)
        # JAX doesn't have mode(), so we estimate or just take mean logits
        # For TPU efficiency, let's keep it simple: Argmax of Sum of Logits
        global_logits = logits.sum(axis=0)  # [3]
        phase_idx = jnp.argmax(global_logits)

        return logits, phase_idx


class StateAccumulator(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, current, accumulated):
        # Gated update
        combined = jnp.concatenate([current, accumulated], axis=-1)
        gate = nn.Dense(self.hidden_dim)(combined)
        gate = nn.sigmoid(gate)

        transformed = nn.Dense(self.hidden_dim)(current)

        new_accumulated = gate * transformed + (1.0 - gate) * accumulated
        return nn.LayerNorm()(new_accumulated)


class LoopEmbedding(nn.Module):
    hidden_dim: int
    max_loops: int

    def setup(self):
        self.embedding = nn.Embed(self.max_loops, self.hidden_dim)
        # Fixed Sinusoidal PE
        position = jnp.arange(0, self.max_loops, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.hidden_dim, 2, dtype=jnp.float32)
            * (-math.log(10000.0) / self.hidden_dim)
        )
        pe = jnp.zeros((self.max_loops, self.hidden_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, loop_idx):
        # loop_idx is scalar (current loop)
        idx = jnp.minimum(loop_idx, self.max_loops - 1)
        learned = self.embedding(idx)
        fixed = self.pe[idx]
        return learned + fixed


class MassivePool(nn.Module):
    config: DPSNRConfig

    def setup(self):
        self.embeddings = self.param(
            "embeddings",
            nn.initializers.normal(stddev=0.02),
            (self.config.pool_size, self.config.pool_dim),
        )

    def __call__(
        self,
        query_hidden: jnp.ndarray,
        k_predicted: jnp.ndarray,
        phase_idx: jnp.ndarray,
    ):
        """
        Retrieves top-k vectors with Phase-aware masking.
        """
        batch, seq, dim = query_hidden.shape
        flat_query = query_hidden.reshape(-1, dim)

        # 1. Compute Scores
        scores = jnp.dot(flat_query, self.embeddings.T)

        # 2. Phase-aware partitioning (Simplified for demo)
        # In full version, we'd mask parts of the pool based on phase_idx
        # For now, we assume the whole pool is available but weighted by phase logic
        # (Original code splits pool into partitions, here we just retrieve from global)

        # 3. Static Top-K
        top_scores, top_indices = jax.lax.top_k(scores, k=self.config.max_k)

        # 4. Dynamic Masking
        iota = jax.lax.iota(jnp.int32, self.config.max_k)
        iota = jnp.expand_dims(iota, 0)
        flat_k = k_predicted.reshape(-1, 1)
        mask = (iota < flat_k).astype(jnp.float32)

        retrieved = self.embeddings[top_indices]
        retrieved = retrieved * mask[:, :, None]

        return retrieved.reshape(batch, seq, self.config.max_k, dim)


# --- Main Model ---


from flax import struct

# ... (ReasoningPhase Enum stays same)


@struct.dataclass
class LoopState:
    hidden: jnp.ndarray
    loop_count: jnp.ndarray  # Changed to array for JAX tracing
    cumulative_halt_prob: jnp.ndarray
    halted_mask: jnp.ndarray
    remainders: jnp.ndarray
    ponder_cost: jnp.ndarray


class DPSNR(nn.Module):
    config: DPSNRConfig

    def setup(self):
        self.embedding = nn.Embed(self.config.vocab_size, self.config.hidden_dim)
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.config.max_seq_len, self.config.hidden_dim),
        )

        self.controller = TinyController(self.config)
        self.pool = MassivePool(self.config)

        # Reasoning Components
        self.halt_predictor = HaltPredictor(
            self.config.hidden_dim, self.config.halt_threshold
        )
        self.phase_classifier = PhaseClassifier(self.config.hidden_dim)
        self.state_accumulator = StateAccumulator(self.config.hidden_dim)
        self.loop_embedding = LoopEmbedding(
            self.config.hidden_dim, self.config.max_loops
        )

        self.integrator = nn.Sequential(
            [
                nn.Dense(self.config.hidden_dim * 2),
                nn.gelu,
                nn.Dense(self.config.hidden_dim),
            ]
        )
        self.k_predictor = nn.Dense(1)
        self.lm_head = nn.Dense(self.config.vocab_size)

        self.dropout = nn.Dropout(rate=self.config.dropout_rate)

    def get_retrieval_k(self, hidden):
        logits = self.k_predictor(hidden)
        k_norm = nn.sigmoid(logits)
        k = self.config.min_k + k_norm * (self.config.max_k - self.config.min_k)
        return jnp.floor(k).astype(jnp.int32)

    def reasoning_step(self, carry: LoopState, _: Any, train: bool = False):
        # Unpack state
        state = carry

        # 1. Loop Embedding
        loop_emb = self.loop_embedding(state.loop_count)
        hidden_input = state.hidden + loop_emb[None, None, :]

        # 2. Controller
        processed = self.controller(hidden_input, train)

        # 3. Phase Classification
        phase_logits, phase_idx = self.phase_classifier(processed)

        # 4. Retrieval
        k_val = self.get_retrieval_k(processed)
        retrieved = self.pool(processed, k_val, phase_idx)

        # 5. Integration
        context = jnp.sum(retrieved, axis=2) / (k_val + 1e-6)
        integrated = self.integrator(jnp.concatenate([processed, context], axis=-1))

        # 6. Accumulate State
        new_hidden = self.state_accumulator(integrated, state.hidden)

        # 7. Halt Prediction (ACT)
        halt_prob = self.halt_predictor(new_hidden)  # [B, T, 1]

        # ACT Logic
        # If cum_prob + current_prob >= 1.0, we halt.
        # But we must respect already halted items.

        still_running = (1.0 - state.halted_mask)[:, :, None]
        new_cumulative = state.cumulative_halt_prob + halt_prob * still_running

        should_halt = new_cumulative >= 1.0

        # Calculate remainder for loss
        # remainder = 1 - cum_prob (before this step)
        remainder = (1.0 - state.cumulative_halt_prob) * should_halt.astype(jnp.float32)

        # Update Mask
        new_halted_mask = jnp.maximum(state.halted_mask, should_halt.squeeze(-1))

        # Update Remainders (add current remainder)
        new_remainders = state.remainders + remainder

        # Update Cumulative (cap at 1.0 if halted)
        # If not halted, it's just new_cumulative. If halted, it's 1.0 (implicitly)
        # But for ponder cost, we track the prob sum.
        final_cumulative = jnp.minimum(new_cumulative, 1.0)

        # 8. Gated State Update
        # If just halted or already halted, we keep the state that triggered the halt
        # Actually, ACT standard: we usually output the state *at* the halt step.
        # So we update everyone, but we'll mask the final output of the loop later.
        # Here we freeze the state in the carry to avoid "drift" after halting.
        update_gate = (1.0 - state.halted_mask)[:, :, None]
        final_hidden_state = (
            state.hidden * (1.0 - update_gate) + new_hidden * update_gate
        )

        new_state = LoopState(
            hidden=final_hidden_state,
            loop_count=state.loop_count + 1,
            cumulative_halt_prob=final_cumulative,
            halted_mask=new_halted_mask,
            remainders=new_remainders,
            ponder_cost=state.ponder_cost,  # accumulated outside
        )

        return new_state, None

    def __call__(self, input_ids, train: bool = False):
        x = self.embedding(input_ids)
        batch, seq = x.shape[:2]

        pos_emb = self.pos_embedding[:, :seq, :]
        x = x + pos_emb

        x = self.dropout(x, deterministic=not train)

        # Init State
        init_state = LoopState(
            hidden=x,
            loop_count=jnp.array(0, dtype=jnp.int32),
            cumulative_halt_prob=jnp.zeros((batch, seq, 1)),
            halted_mask=jnp.zeros((batch, seq), dtype=jnp.float32),
            remainders=jnp.zeros((batch, seq, 1)),
            ponder_cost=jnp.zeros((1,)),
        )

        # Init Params Hack
        _ = self.reasoning_step(init_state, None, train=train)

        # Scan
        def scan_step(carry, x):
            return self.reasoning_step(carry, x, train=train)

        final_state, _ = jax.lax.scan(
            scan_step, init_state, None, length=self.config.max_loops
        )

        # Compute Ponder Cost (Avg loops + Remainders)
        # ACT Loss = Mean(Cumulative Prob) + Mean(Remainders)
        ponder_loss = jnp.mean(final_state.cumulative_halt_prob) + jnp.mean(
            final_state.remainders
        )

        logits = self.lm_head(final_state.hidden)

        return {
            "logits": logits,
            "ponder_loss": ponder_loss,
            "loops": final_state.cumulative_halt_prob.sum(
                axis=-1
            ).mean(),  # Approximate loop count
        }
