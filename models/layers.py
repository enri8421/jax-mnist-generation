"""Shared image and timestep layers. Images use NHWC layout."""

import math

import jax
import jax.numpy as jnp
from flax import nnx

# B: batch; H, W: image size; C: channels; T: timestep embedding width.


def resize(images: jax.Array, size: tuple[int, int]) -> jax.Array:
    """Resize spatial axes without mixing examples or channels."""
    # (B, *size, C)
    return jax.image.resize(images, (images.shape[0], *size, images.shape[-1]), "linear")


def with_coordinates(images: jax.Array) -> jax.Array:
    """Append fixed spatial coordinates so constant latents can describe shapes."""
    # images: (B, H, W, C); coordinates: (B, H, W, 2), in [-1, 1].
    y, x = jnp.meshgrid(
        jnp.linspace(-1, 1, images.shape[1]),
        jnp.linspace(-1, 1, images.shape[2]),
        indexing="ij",
    )
    coordinates = jnp.stack((x, y), axis=-1)
    coordinates = jnp.broadcast_to(coordinates, (images.shape[0], *coordinates.shape))
    return jnp.concatenate((images, coordinates), axis=-1)


class TimeEmbedding(nnx.Module):
    """Sinusoidal timesteps followed by a small MLP."""

    def __init__(self, width: int = 128, *, rngs: nnx.Rngs):
        self.width = width
        self.hidden = nnx.Linear(width, width, rngs=rngs)
        self.output = nnx.Linear(width, width, rngs=rngs)

    def __call__(self, timesteps: jax.Array) -> jax.Array:
        # timesteps: (B,)
        half = self.width // 2
        frequencies = jnp.exp(-math.log(10_000) * jnp.arange(half) / (half - 1))  # (T/2,)
        angles = timesteps.astype(jnp.float32)[:, None] * frequencies[None, :]  # (B, T/2)
        embedding = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)  # (B, T)
        return self.output(nnx.silu(self.hidden(embedding)))  # (B, width)


class ResidualBlock(nnx.Module):
    """Two convolutions with timestep conditioning and a skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_width: int = 128,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.GroupNorm(in_channels, num_groups=8, rngs=rngs)
        self.conv1 = nnx.Conv(in_channels, out_channels, (3, 3), rngs=rngs)
        self.time = nnx.Linear(time_width, out_channels, rngs=rngs)
        self.norm2 = nnx.GroupNorm(out_channels, num_groups=8, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, (3, 3), rngs=rngs)
        self.skip = (
            nnx.Conv(in_channels, out_channels, (1, 1), rngs=rngs)
            if in_channels != out_channels
            else None
        )

    def __call__(self, images: jax.Array, time: jax.Array) -> jax.Array:
        # images: (B, H, W, C_in); time: (B, T)
        hidden = self.conv1(nnx.silu(self.norm1(images)))  # (B, H, W, C_out)
        hidden = hidden + self.time(nnx.silu(time))[:, None, None, :]  # add (B, 1, 1, C_out)
        hidden = self.conv2(nnx.silu(self.norm2(hidden)))  # (B, H, W, C_out)
        skip = images if self.skip is None else self.skip(images)  # (B, H, W, C_out)
        return hidden + skip  # (B, H, W, C_out)
