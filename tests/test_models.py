"""Model contracts and differentiability, without downloading or training MNIST."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from models import diffusion, layers, tokenizer, transformer


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = jax.random.uniform(jax.random.key(0), (2, 28, 28, 1))
        self.labels = jnp.array([2, 7], dtype=jnp.int32)
        self.timesteps = jnp.array([0, 50], dtype=jnp.int32)

    def test_decoder_coordinates_preserve_latents_and_locate_pixels(self) -> None:
        latents = jnp.ones((2, 7, 7, 32))
        located = jax.jit(layers.with_coordinates)(latents)
        self.assertEqual(located.shape, (2, 7, 7, 34))
        np.testing.assert_array_equal(located[..., :32], latents)
        np.testing.assert_allclose(located[0, 0, :, -2], np.linspace(-1, 1, 7), atol=1e-6)
        np.testing.assert_allclose(located[0, :, 0, -1], np.linspace(-1, 1, 7), atol=1e-6)
        np.testing.assert_array_equal(located[0, :, :, -2:], located[1, :, :, -2:])
        gradient = jax.grad(lambda values: jnp.sum(layers.with_coordinates(values)))(latents)
        np.testing.assert_array_equal(gradient, 1)

    def test_residual_codes_and_prefix_decoding(self) -> None:
        codec = tokenizer.Tokenizer(rngs=nnx.Rngs(1))
        reconstructed, latents, quantized = nnx.jit(lambda m, x: m(x))(codec, self.images)
        codes = codec.encode(self.images)
        self.assertEqual(latents.shape, (2, 7, 7, 32))
        self.assertEqual([grid.shape for grid in codes], [(2, 1, 1), (2, 3, 3), (2, 7, 7)])
        np.testing.assert_allclose(quantized, codec.embed(codes), atol=1e-5)
        np.testing.assert_allclose(reconstructed, codec.decode(codes), atol=1e-5)
        for level, grid in enumerate(codes):
            self.assertTrue(np.issubdtype(grid.dtype, np.integer))
            self.assertTrue(bool(jnp.all((grid >= 0) & (grid < 64))))
            self.assertEqual(codec.decode(codes[: level + 1]).shape, self.images.shape)
        self.assertTrue(bool(jnp.all((reconstructed >= 0) & (reconstructed <= 1))))

    def test_quantization_selects_nearest_residual_codes(self) -> None:
        codec = tokenizer.Tokenizer(rngs=nnx.Rngs(1))
        # Constant residuals select a known code independently at each scale.
        for codebook in codec.codebooks:
            codebook.embedding[...] = jnp.full((64, 32), 10.0).at[0].set(0.0).at[1].set(1.0)
        latents = jnp.full((1, 7, 7, 32), 1.25)
        quantized, codes = codec.quantize(latents)
        np.testing.assert_allclose(quantized, 1.0)
        np.testing.assert_array_equal(codes[0], 1)
        for grid in codes[1:]:
            np.testing.assert_array_equal(grid, 0)
        for prefix_length in range(1, 4):
            np.testing.assert_allclose(codec.embed(codes[:prefix_length]), 1.0)

    def test_transformer_shapes_and_conditioning(self) -> None:
        model = transformer.Transformer(rngs=nnx.Rngs(2))
        context = jnp.zeros((2, 7, 7, 32))
        for level, size in enumerate(tokenizer.SCALES):
            forward = nnx.jit(lambda m, x, labels: m(x, labels, level))
            logits = forward(model, context, self.labels)
            self.assertEqual(logits.shape, (2, size * size, 64))
            repeated = forward(model, context, self.labels)
            np.testing.assert_allclose(logits, repeated, atol=1e-6)
            relabelled = forward(model, context, (self.labels + 1) % 10)
            changed_context = forward(model, context + 0.5, self.labels)
            self.assertGreater(float(jnp.max(jnp.abs(logits - relabelled))), 1e-6)
            self.assertGreater(float(jnp.max(jnp.abs(logits - changed_context))), 1e-6)

    def test_joint_positions_distinguish_locations_and_use_disjoint_rows(self) -> None:
        model = transformer.Transformer(rngs=nnx.Rngs(2))
        context = jnp.zeros((2, 7, 7, 32))
        logits = model(context, self.labels, 1)
        self.assertGreater(float(jnp.max(jnp.abs(logits[:, 0] - logits[:, -1]))), 1e-6)
        for level, (start, stop) in enumerate(((0, 1), (1, 10), (10, 59))):
            gradients = nnx.grad(lambda m: jnp.mean(m(context, self.labels, level) ** 2))(model)
            rows = np.asarray(gradients.positions.embedding)
            self.assertEqual(rows.shape, (59, 128))
            self.assertTrue(np.any(rows[start:stop] != 0))
            unused = np.concatenate((rows[:start], rows[stop:]))
            np.testing.assert_array_equal(unused, 0)

    def test_denoiser_uses_timesteps(self) -> None:
        model = diffusion.Denoiser(rngs=nnx.Rngs(4))
        forward = nnx.jit(lambda model, times: model(self.images * 2 - 1, times, self.labels))
        first = forward(model, jnp.zeros(2, dtype=jnp.int32))
        second = forward(model, jnp.full((2,), 50, dtype=jnp.int32))
        self.assertEqual(first.shape, self.images.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(first))))
        self.assertGreater(float(jnp.max(jnp.abs(first - second))), 1e-6)

    def test_denoiser_uses_digit_and_unconditional_labels(self) -> None:
        model = diffusion.Denoiser(rngs=nnx.Rngs(0))
        forward = nnx.jit(lambda model, labels: model(self.images, self.timesteps, labels))
        labelled = forward(model, self.labels)
        changed = forward(model, (self.labels + 1) % 10)
        unconditional = forward(model, jnp.full_like(self.labels, 10))
        self.assertGreater(float(jnp.max(jnp.abs(labelled - changed))), 1e-6)
        self.assertGreater(float(jnp.max(jnp.abs(labelled - unconditional))), 1e-6)

    def test_compiled_model_gradients(self) -> None:
        models = [
            tokenizer.Tokenizer(rngs=nnx.Rngs(1)),
            transformer.Transformer(rngs=nnx.Rngs(2)),
            diffusion.Denoiser(rngs=nnx.Rngs(3)),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):

                def objective(m: nnx.Module) -> jax.Array:
                    if isinstance(m, tokenizer.Tokenizer):
                        reconstruction, _, quantized = m(self.images)
                        return jnp.mean(reconstruction**2) + jnp.mean(quantized**2)
                    if isinstance(m, transformer.Transformer):
                        context = jnp.zeros((2, 7, 7, 32))
                        return jnp.mean(m(context, self.labels, 2) ** 2)
                    assert isinstance(m, diffusion.Denoiser)
                    prediction = m(self.images * 2 - 1, self.timesteps, self.labels)
                    return jnp.mean(prediction**2)

                value, gradients = nnx.jit(nnx.value_and_grad(objective))(model)
                leaves = jax.tree.leaves(gradients)
                self.assertTrue(np.isfinite(float(value)))
                self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in leaves))
                self.assertTrue(any(np.any(np.asarray(x) != 0) for x in leaves))
                if isinstance(model, tokenizer.Tokenizer):
                    # The straight-through path reaches the encoder; lookups reach codebooks.
                    for part in (gradients.encoder, gradients.codebooks):
                        self.assertTrue(
                            any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(part)),
                        )


if __name__ == "__main__":
    unittest.main()
