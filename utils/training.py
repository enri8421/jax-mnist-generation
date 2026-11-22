"""Three small trainers and one shared epoch loop."""

import collections.abc
import enum
import math
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from models import diffusion, tokenizer, transformer
from utils import checkpoints, data, noise


class ModelKind(enum.StrEnum):
    TOKENIZER = "tokenizer"
    TRANSFORMER = "transformer"
    DENOISER = "denoiser"


class Trainer(nnx.Module):
    """Own the trainable model and optimizer; share one compiled update."""

    loss: collections.abc.Callable[..., jax.Array]

    def __init__(self, model: nnx.Module, learning_rate: float):
        self.model = model
        self.optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    def prepare(
        self,
        arrays: data.Arrays,
        *,
        weights: pathlib.Path,
        batch_size: int,
    ) -> data.Arrays:
        return arrays

    def step(self, batch: data.Batch, key: jax.Array) -> jax.Array:
        return _update(self, batch, key)


@nnx.jit
def _update(trainer: Trainer, batch: data.Batch, key: jax.Array) -> jax.Array:
    def loss(model: nnx.Module) -> jax.Array:
        return trainer.loss(model, batch, key)

    value, gradients = nnx.value_and_grad(loss)(trainer.model)
    trainer.optimizer.update(trainer.model, gradients)
    return value


class TokenizerTrainer(Trainer):
    """Reconstruct random residual prefixes with VQ and commitment losses."""

    model: tokenizer.Tokenizer

    def __init__(self, seed: int, learning_rate: float):
        super().__init__(tokenizer.Tokenizer(rngs=nnx.Rngs(seed)), learning_rate)

    def loss(self, model: tokenizer.Tokenizer, batch: data.Batch, key: jax.Array) -> jax.Array:
        images, _ = batch
        keep = jax.random.randint(key, (images.shape[0], 1, 1, 1), 1, 4)
        latents = model.encoder(images)  # (B, 7, 7, 32)
        _, codes = model.quantize(latents)
        quantized = jnp.zeros_like(latents)
        for level, grid in enumerate(codes):
            quantized = quantized + (level < keep) * model.embed_level(grid, level)
        straight_through = latents + jax.lax.stop_gradient(quantized - latents)
        reconstruction = model.decoder(straight_through)  # (B, 28, 28, 1)
        return (
            jnp.mean((reconstruction - images) ** 2)
            + jnp.mean((quantized - jax.lax.stop_gradient(latents)) ** 2)
            + 0.25 * jnp.mean((latents - jax.lax.stop_gradient(quantized)) ** 2)
        )


class TransformerTrainer(Trainer):
    """Own the frozen tokenizer and predict each scale's clean codes."""

    model: transformer.Transformer

    def __init__(self, seed: int, learning_rate: float):
        super().__init__(transformer.Transformer(rngs=nnx.Rngs(seed)), learning_rate)
        self.codec = tokenizer.Tokenizer(rngs=nnx.Rngs(0))

    def prepare(
        self,
        arrays: data.Arrays,
        *,
        weights: pathlib.Path,
        batch_size: int,
    ) -> data.Arrays:
        images, labels = arrays
        checkpoints.load(self.codec, weights / "tokenizer.msgpack")
        print("Encoding training images once with the frozen tokenizer", flush=True)
        return (*data.encode(self.codec, images, batch_size), labels)

    def loss(self, model: transformer.Transformer, batch: data.Batch, key: jax.Array) -> jax.Array:
        codes, labels = batch[:-1], batch[-1]
        context = jnp.zeros((labels.shape[0], 7, 7, 32))
        losses = []
        for level, grid in enumerate(codes):
            logits = model(context, labels, level)  # (B, s*s, 64)
            targets = grid.reshape(grid.shape[0], -1)
            cross_entropy = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            level_loss = jnp.mean(cross_entropy)
            losses.append(level_loss)
            key, mask_key, code_key = jax.random.split(key, 3)
            corrupted = jnp.where(
                jax.random.bernoulli(mask_key, 0.1, grid.shape),
                jax.random.randint(code_key, grid.shape, 0, 64),
                grid,
            )
            context = context + jax.lax.stop_gradient(self.codec.embed_level(corrupted, level))
        return jnp.mean(jnp.stack(losses))


class DenoiserTrainer(Trainer):
    """Predict v with 10% label dropout for classifier-free guidance."""

    model: diffusion.Denoiser

    def __init__(self, seed: int, learning_rate: float):
        super().__init__(diffusion.Denoiser(rngs=nnx.Rngs(seed)), learning_rate)

    def loss(self, model: diffusion.Denoiser, batch: data.Batch, key: jax.Array) -> jax.Array:
        images, labels = batch
        noise_key, label_key = jax.random.split(key)
        clean = images * 2 - 1
        noisy, epsilon, timesteps = noise.sample(noise_key, clean)
        labels = jnp.where(jax.random.bernoulli(label_key, 0.1, labels.shape), 10, labels)
        prediction = model(noisy, timesteps, labels)
        target = noise.velocity(clean, epsilon, timesteps)
        return jnp.mean((prediction - target) ** 2)


def create(kind: ModelKind, *, seed: int, learning_rate: float) -> Trainer:
    """Choose one trainer; model-specific setup stays inside its class."""
    trainer_type = {
        ModelKind.TOKENIZER: TokenizerTrainer,
        ModelKind.TRANSFORMER: TransformerTrainer,
        ModelKind.DENOISER: DenoiserTrainer,
    }[kind]
    return trainer_type(seed, learning_rate)


def fit(
    trainer: Trainer,
    arrays: data.Arrays,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> None:
    """Shuffle, batch, update, and report a sample-weighted loss per epoch."""
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    for epoch in range(epochs):
        start = time.perf_counter()
        losses, sizes = [], []
        for batch in data.batches(arrays, batch_size, rng):
            key, step_key = jax.random.split(key)
            losses.append(trainer.step(batch, step_key))
            sizes.append(batch[0].shape[0])
        average = float(jnp.average(jnp.stack(losses), weights=jnp.array(sizes)))
        if not math.isfinite(average):
            raise FloatingPointError("Non-finite training loss")
        elapsed = time.perf_counter() - start
        print(
            f"{type(trainer.model).__name__}: epoch {epoch + 1}/{epochs}, "
            f"loss={average:.5f}, {elapsed:.1f}s",
            flush=True,
        )
