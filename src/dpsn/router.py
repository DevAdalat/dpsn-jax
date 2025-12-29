import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any


class Router(nn.Module):
    num_memory_slots: int
    min_k: int
    max_k: int
    router_dim: int = 0  # Default to 0, which means auto-calculate

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: Input embeddings (batch, seq_len, d_model) or (batch, d_model)
            training: Whether in training mode (adds noise)

        Returns:
            indices: Selected indices (batch, ..., max_k)
            weights: Aggregation weights (batch, ..., max_k)
            budget: Calculated budget k (batch, ..., 1) - purely informational/masking
            aux_loss: Auxiliary loss scalar (for load balancing)
        """
        # x shape might be (batch, seq, d_model) or just (batch, d_model)
        # We operate on the last dimension

        # 1. Complexity Analysis
        # c = sigmoid(W_c x + b_c)
        complexity_logits = nn.Dense(features=1, name="complexity_net")(x)
        complexity_score = nn.sigmoid(complexity_logits)  # (batch, ..., 1)

        # Calculate budget k
        # k = floor(k_min + (k_max - k_min) * c^2)
        # Note: In JAX, we can't easily have dynamic array sizes for the next step.
        # We will select max_k indices always, and mask the ones exceeding the budget.

        k_range = self.max_k - self.min_k
        budget_float = self.min_k + k_range * (complexity_score**2)
        budget = jnp.floor(budget_float).astype(jnp.int32)

        # 2. Index Selection
        # S = ReLU(W1 x) W2
        # To avoid O(M) in a real large scale system, we'd need hierarchical routing.
        # For this implementation (M=20k), O(M) is fine.
        # Use a bottleneck dimension for the router hidden layer to keep it lightweight.

        # Determine router dimension
        r_dim = self.router_dim
        if r_dim <= 0:
            r_dim = max(32, x.shape[-1] // 8)

        hidden = nn.Dense(features=r_dim, name="router_hidden")(x)
        hidden = nn.relu(hidden)
        scores = nn.Dense(features=self.num_memory_slots, name="router_scores")(hidden)

        # Calculate Load Balancing Loss (Auxiliary Loss)
        # We want to encourage uniform usage of the slots.
        # L_balance = num_slots * sum(P_i * f_i)
        # P_i: Average probability assigned to slot i (from softmax of scores)
        # f_i: Fraction of tokens that selected slot i

        # Softmax over all scores (not just top-k) to get the router's "intent"
        router_probs = nn.softmax(scores, axis=-1)
        # Mean probability per slot across the batch
        # Flatten batch dimensions: (batch, seq, slots) -> (batch*seq, slots)
        avg_probs = jnp.mean(router_probs.reshape(-1, self.num_memory_slots), axis=0)

        if training:
            # Add Gumbel noise or similar
            noise = jax.random.uniform(self.make_rng("noise"), scores.shape)
            # Gumbel = -log(-log(uniform))
            gumbel_noise = -jnp.log(-jnp.log(noise + 1e-10) + 1e-10)
            scores_with_noise = scores + gumbel_noise
        else:
            scores_with_noise = scores

        # TopK selection
        # We always select max_k top indices.
        # We will create a mask based on the dynamic budget.

        top_k_values, top_k_indices = jax.lax.top_k(scores_with_noise, self.max_k)

        # Create mask: 1 if rank < budget, 0 otherwise
        # ranges: [0, 1, ..., max_k-1]
        ranks = jnp.arange(self.max_k)
        # ranks shape: (max_k,)
        # budget shape: (batch, ..., 1)
        # Broadcast comparison
        mask = ranks < budget  # (batch, ..., max_k)

        # Softmax over selected scores for weighting?
        # Paper: "results are weighted by the router's softmax probability"
        # We softmax the top_k scores.
        # Masking before softmax? Or after?
        # Usually softmax over the active set.

        # Masking logits for softmax: set masked values to -inf
        masked_logits = jnp.where(mask, top_k_values, -1e9)
        weights = nn.softmax(masked_logits, axis=-1)

        # Zero out weights that were masked (just in case softmax didn't kill them enough, or to be explicit)
        weights = weights * mask

        # Calculate f_i (fraction of selection)
        # We use scatter_add to avoid OOM from materializing large one_hot tensor
        flat_indices = top_k_indices.reshape(-1)
        flat_mask = mask.reshape(-1).astype(jnp.float32)

        counts = jnp.zeros((self.num_memory_slots,), dtype=jnp.float32)
        counts = counts.at[flat_indices].add(flat_mask)

        # Normalize: sum(mask for slot i) / (batch * seq)
        # total_steps (batch * seq) is total elements / max_k
        total_steps = top_k_indices.size / self.max_k
        avg_selection = counts / total_steps

        # Load Balancing Loss: num_slots * sum(avg_probs * avg_selection)
        # We use a coefficient (alpha) in the main training loop, here we just return the raw term.
        # Ideally, P_i and f_i should be close to 1/N.
        # If perfect balance: sum((1/N) * (1/N)) * N = N * (1/N^2) * N = 1.
        # If collapse (1 slot used): sum(1 * 1) + 0... * N = N.
        load_balancing_loss = self.num_memory_slots * jnp.sum(avg_probs * avg_selection)

        # Z-Loss: Log(sum(exp(logits))^2) -> penalty for large logits
        # log_sum_exp_logits = jax.nn.logsumexp(scores, axis=-1)
        # z_loss = jnp.mean(jnp.square(log_sum_exp_logits))

        # Combine auxiliary losses (we can return them separately or summed)
        # Let's return just the load balancing loss for now as it's the most critical "Lazy Librarian" fix.
        # We can add Z-loss later if needed.

        aux_loss = load_balancing_loss

        return top_k_indices, weights, budget, aux_loss
