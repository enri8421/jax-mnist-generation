"""Label-conditioned next-scale prediction, parallel within each residual grid."""

import jax
import jax.numpy as jnp
from flax import nnx

from models import layers, tokenizer

# B: batch; s: current scale; N=s*s; D=32: latent channels; C=128: model width.


class TransformerBlock(nnx.Module):
    """Pre-norm attention and a feedforward residual."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(128, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=4,
            in_features=128,
            decode=False,
            deterministic=True,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(128, rngs=rngs)
        self.hidden = nnx.Linear(128, 256, rngs=rngs)
        self.output = nnx.Linear(256, 128, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        tokens = tokens + self.attention(self.norm1(tokens))  # (B, N, C)
        return tokens + self.output(nnx.gelu(self.hidden(self.norm2(tokens))))  # C -> 2C -> C


class Transformer(nnx.Module):
    """Predict a grid from the digit and cumulative earlier-scale latents."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.input = nnx.Linear(32, 128, rngs=rngs)
        self.labels = nnx.Embed(10, 128, rngs=rngs)
        self.positions = nnx.Embed(59, 128, rngs=rngs)  # 1 + 9 + 49 positions
        self.blocks = nnx.List([TransformerBlock(rngs=rngs) for _ in range(3)])
        self.norm = nnx.LayerNorm(128, rngs=rngs)
        self.head = nnx.Linear(128, 64, rngs=rngs)

    def __call__(self, context: jax.Array, labels: jax.Array, level: int) -> jax.Array:
        """Return code logits for one level; construct context outside the model."""
        # context: (B, 7, 7, D); labels: (B,)
        size = tokenizer.SCALES[level]
        # (B, N, D)
        context = layers.resize(context, (size, size)).reshape(context.shape[0], size * size, 32)

        tokens = self.input(context) + self.labels(labels)[:, None, :]  # (B, N, C)

        position_ids = (0, 1, 10)[level] + jnp.arange(size * size)  # (N,), distinct IDs per scale
        tokens = tokens + self.positions(position_ids)[None, :, :]  # add (1, N, C)
        for block in self.blocks:
            tokens = block(tokens)  # (B, N, C)
        return self.head(self.norm(tokens))  # (B, N, 64)
