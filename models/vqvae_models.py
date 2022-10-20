import jax.numpy as jnp
import jax
import haiku as hk
import optax 
from flax import struct
from collections import namedtuple


class QuantizedCodebook(hk.Module):  
    def __init__(self, K, D, beta, name):
        super().__init__(name)
        self.K = K
        self.D = D
        self.beta = beta

        initializer = hk.initializers.VarianceScaling(distribution="uniform")
        self.codebook = hk.get_parameter("codebook", (self.K, self.D), init=initializer)

    def __call__(self, inputs):
        flattened = jnp.reshape(inputs, (-1, self.D))
        flattened_sqr = jnp.sum(flattened**2, axis=-1, keepdims=True)
        codeboook_sqr = jnp.sum(self.codebook**2, axis=-1, keepdims=True).T
        distances = flattened_sqr - 2 * (flattened @ self.codebook.T) + codeboook_sqr

        encoding_indices = jnp.reshape(
            jnp.argmin(distances, axis=-1), inputs.shape[:-1]
        )

        quantize = self.codebook[encoding_indices]

        encoding_loss = jnp.mean((jax.lax.stop_gradient(quantize) - inputs) ** 2)
        commit_loss = jnp.mean((quantize - jax.lax.stop_gradient(inputs)) ** 2)
        loss = encoding_loss + self.beta * commit_loss
        quantize = inputs + jax.lax.stop_gradient(quantize - inputs)

        return loss, quantize, encoding_indices

    def embed(self, indices):
        outshape = indices.shape + (self.D,)
        x = self.codebook[indices].reshape(outshape)
        return x


class ResBlock(hk.Module):
    def __init__(self, dim, kernel_size, name = None):
        super().__init__(name)
        self.dim = dim
        self.kernel_size = kernel_size

    def __call__(self, x, is_training):
        res = hk.Conv2D(self.dim, self.kernel_size)(x)
        res = hk.BatchNorm(True, True, 0.9)(res, is_training)
        res = jax.nn.relu(res)
        res = hk.Conv2D(self.dim, self.kernel_size)(res)
        res = hk.BatchNorm(True, True, 0.9)(res, is_training)
        x += res
        x = jax.nn.relu(x)
        return x

class CnnEncoder(hk.Module):
    def __init__(self, out_channels, downscale_level, res_layers, name, kernel_size = 5):
        super().__init__(name)
        self.out_channels = out_channels
        self.downscale_level = downscale_level
        self.res_layers = res_layers
        self.kernel_size = kernel_size

    def __call__(self, x, is_training):
        for i in range(self.downscale_level - 1, -1, -1):
            num_channels = self.out_channels // (2**i)
            x = hk.Conv2D(num_channels, self.kernel_size, stride=2)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = jax.nn.relu(x)
            for _ in range(self.res_layers):
                x = ResBlock(num_channels, self.kernel_size)(x, is_training)
        return x

class CnnDecoder(hk.Module):
    def __init__(self, in_channels, upscale_level, res_layers, name, kernel_size = 5):
        super().__init__(name)
        self.in_channels = in_channels
        self.upscale_level = upscale_level
        self.res_layers = res_layers
        self.kernel_size = kernel_size

    def __call__(self, x, is_training):
        for i in range(self.upscale_level - 1):
            num_channels = self.in_channels // (2**i)
            x = hk.Conv2DTranspose(num_channels, self.kernel_size, stride=2)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = jax.nn.relu(x)
            for _ in range(self.res_layers):
                x = ResBlock(num_channels, self.kernel_size)(x, is_training)
        x = hk.Conv2DTranspose(1, self.kernel_size, stride=2)(x)

        x = jax.nn.sigmoid(x)
        return x


@struct.dataclass
class Params:
    weights: dict
    opt_state: tuple
    state: dict

VQVAE_model = namedtuple('Nets_and_Opts', 'encode decode quantize embded optimizer')

def build_model(key, input_shape, latent_dim, num_embeddings, compression_level, res_layers, lr, weight_decay, beta):
    key_model, key_quantizer = jax.random.split(key)

    def f():

        encoder = CnnEncoder(latent_dim, compression_level, res_layers, "encoder")
        decoder = CnnDecoder(latent_dim, compression_level, res_layers, "decoder")
        quantizer = QuantizedCodebook(num_embeddings, latent_dim, 0.25, "quantizer")
        def encode(x, is_training):
            return encoder(x, is_training)

        def decode(x, is_training):
            return decoder(x, is_training)

        def quantize(x):
            return quantizer(x)

        def embed(x):
            return quantizer.embed(x)
        
        def init(x, is_training):
            encodings = encode(x, is_training)
            loss, q, idx = quantize(encodings)
            x_pred = decode(q, is_training)
            z_q = embed(idx)
            return x_pred, z_q
        return init, (encode, decode, quantize, embed)

    trasformed = hk.multi_transform_with_state(f)

    weights, state = trasformed.init(key_model, jnp.zeros(((1,) + input_shape)), True)

    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(weights)
    nets_and_opts = VQVAE_model(*trasformed.apply, optimizer)

    params = Params(weights, opt_state, state)

    return params, nets_and_opts