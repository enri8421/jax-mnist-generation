"""Residual VQ autoencoder for 28x28 digits, inspired by arXiv:2610.14162."""

import jax
import jax.numpy as jnp
from flax import nnx

from models import layers

# B: batch; s: current scale; D=32: latent channels; K=64: codebook size.
SCALES = (1, 3, 7)


class Tokenizer(nnx.Module):
    """Encode digits into three residual grids and decode their sum."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.encoder = nnx.Sequential(
            nnx.Conv(1, 32, (3, 3), strides=2, rngs=rngs),  # (B, 14, 14, 32)
            nnx.relu,
            nnx.Conv(32, 64, (3, 3), strides=2, rngs=rngs),  # (B, 7, 7, 64)
            nnx.relu,
            nnx.Conv(64, 32, (1, 1), rngs=rngs),  # (B, 7, 7, D)
        )
        self.codebooks = nnx.List([nnx.Embed(64, 32, rngs=rngs) for _ in SCALES])
        self.decoder = nnx.Sequential(
            layers.with_coordinates,  # (B, 7, 7, D + 2)
            nnx.ConvTranspose(34, 64, (4, 4), strides=2, rngs=rngs),  # (B, 14, 14, 64)
            nnx.relu,
            nnx.ConvTranspose(64, 32, (4, 4), strides=2, rngs=rngs),  # (B, 28, 28, 32)
            nnx.relu,
            nnx.Conv(32, 1, (3, 3), rngs=rngs),  # (B, 28, 28, 1)
            nnx.sigmoid,
        )

    def __call__(self, images: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return reconstructed images, encoder latents, and quantized latents."""
        latents = self.encoder(images)  # (B, 7, 7, D)
        quantized, _ = self.quantize(latents)  # (B, 7, 7, D)
        straight_through = latents + jax.lax.stop_gradient(quantized - latents)
        return self.decoder(straight_through), latents, quantized

    def encode(self, images: jax.Array) -> tuple[jax.Array, ...]:
        """Return code grids (B, 1, 1), (B, 3, 3), and (B, 7, 7)."""
        return self.quantize(self.encoder(images))[1]

    def decode(self, codes: tuple[jax.Array, ...]) -> jax.Array:
        """Decode a full hierarchy or a nonempty coarse prefix."""
        return self.decoder(self.embed(codes))  # (B, 28, 28, 1)

    def quantize(self, latents: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        """Quantize each scale's residual and sum its expanded contribution."""
        quantized = jnp.zeros_like(latents)  # (B, 7, 7, D)
        codes = []
        for level, size in enumerate(SCALES):
            residual = layers.resize(latents - quantized, (size, size))  # (B, s, s, D)
            codebook = self.codebooks[level].embedding[...]  # (K, D)
            distances = (
                jnp.sum(residual**2, axis=-1, keepdims=True)
                - 2 * residual @ codebook.T
                + jnp.sum(codebook**2, axis=-1)
            )  # (B, s, s, K)
            grid = jnp.argmin(distances, axis=-1)  # (B, s, s)
            codes.append(grid)
            quantized = quantized + self.embed_level(grid, level)  # (B, 7, 7, D)
        return quantized, tuple(codes)

    def embed_level(self, codes: jax.Array, level: int) -> jax.Array:
        """Expand one code grid to the full latent resolution."""
        return layers.resize(self.codebooks[level](codes), (7, 7))  # (B, 7, 7, D)

    def embed(self, codes: tuple[jax.Array, ...]) -> jax.Array:
        """Sum a full hierarchy or a nonempty prefix."""
        latents = self.embed_level(codes[0], 0)  # (B, 7, 7, D)
        for level, grid in enumerate(codes[1:], start=1):
            latents = latents + self.embed_level(grid, level)
        return latents
