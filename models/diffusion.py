"""A small timestep-conditioned U-Net predicting v for label-conditioned MNIST generation."""

import jax
import jax.numpy as jnp
from flax import nnx

from models import layers

# B: batch; spatial sizes below use 28x28 inputs.


class Denoiser(nnx.Module):
    """Two downsampling stages and matching skip-connected upsampling stages."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.time = layers.TimeEmbedding(rngs=rngs)
        self.labels = nnx.Embed(11, 128, rngs=rngs)  # 0–9: digits; 10: unconditional.
        self.input = nnx.Conv(1, 32, (3, 3), rngs=rngs)
        self.down1 = layers.ResidualBlock(32, 32, rngs=rngs)
        self.pool1 = nnx.Conv(32, 64, (3, 3), strides=2, rngs=rngs)
        self.down2 = layers.ResidualBlock(64, 64, rngs=rngs)
        self.pool2 = nnx.Conv(64, 128, (3, 3), strides=2, rngs=rngs)
        self.middle = layers.ResidualBlock(128, 128, rngs=rngs)
        self.up2 = layers.ResidualBlock(128 + 64, 64, rngs=rngs)
        self.up1 = layers.ResidualBlock(64 + 32, 32, rngs=rngs)
        self.norm = nnx.GroupNorm(32, num_groups=8, rngs=rngs)
        self.output = nnx.Conv(32, 1, (3, 3), rngs=rngs)

    def __call__(self, images: jax.Array, timesteps: jax.Array, labels: jax.Array) -> jax.Array:
        """Predict v (B, H, W, 1), conditioned on timesteps and digit labels."""
        # images: (B, 28, 28, 1); timesteps and labels: (B,)
        time = self.time(timesteps) + self.labels(labels)  # (B, 128)
        skip1 = self.down1(self.input(images), time)  # (B, 28, 28, 32)
        skip2 = self.down2(self.pool1(skip1), time)  # (B, 14, 14, 64)
        hidden = self.middle(self.pool2(skip2), time)  # (B, 7, 7, 128)
        hidden = layers.resize(hidden, (skip2.shape[1], skip2.shape[2]))  # (B, 14, 14, 128)
        hidden = self.up2(jnp.concatenate((hidden, skip2), axis=-1), time)  # 192 -> 64 channels
        hidden = layers.resize(hidden, (skip1.shape[1], skip1.shape[2]))  # (B, 28, 28, 64)
        hidden = self.up1(jnp.concatenate((hidden, skip1), axis=-1), time)  # 96 -> 32 channels
        return self.output(nnx.silu(self.norm(hidden)))  # (B, 28, 28, 1)
