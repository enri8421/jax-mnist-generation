import jax.numpy as jnp
import jax
import haiku as hk
import optax 
from flax import struct
from collections import namedtuple
import math


class TimeEmbedding(hk.Module):

    def __init__(self, dim, name):
        super().__init__(name)
        self.dim = dim
        half_dim = dim // 2
        self.fixed_emb = jnp.exp(jnp.arange(half_dim) * - (math.log(10000) / (half_dim - 1)))

    def __call__(self, x):
        time_dim = self.dim*4
        emb = x[:, None] * self.fixed_emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        res = hk.nets.MLP([time_dim, time_dim], activation = jax.nn.gelu)(emb)
        return res


class Block(hk.Module):
    def __init__(self, dim, groups, name):
        super().__init__(name)

        self.forward = hk.Sequential((hk.Conv2D(dim, 3), hk.GroupNorm(groups), jax.nn.silu))

    def __call__(self, x):
        return self.forward(x)

class ResnetBlock(hk.Module):
    def __init__(self, dim, groups, name):
        super().__init__(name)
        self.name = name
        self.dim = dim
        self.groups = groups

    def __call__(self, x, t = None):
        res = Block(self.dim, self.groups, self.name + "block1")(x)
        if t is not None:
            t = jax.nn.silu(t)
            t = hk.Linear(self.dim)(t)
            y = jnp.expand_dims(jnp.expand_dims(t, 1), 1) + res
        res = Block(self.dim, self.groups, self.name + "block2")(res)
        conv = hk.Conv2D(self.dim, 1)(x)
        return res + conv

class ImageMultiHeadAttention(hk.Module):
    def __init__(self, dim, groups, name):
        super().__init__(name)
        self.dim = dim
        self.groups = groups
        initializer = hk.initializers.VarianceScaling(0.1)
        self.att_layer = hk.MultiHeadAttention(groups, dim//groups,w_init = initializer, model_size = dim)

    def __call__(self, x):
        b, h, w, c = x.shape
        x = jnp.reshape(x, (b, h*w, c))
        res = self.att_layer(x, x, x)
        res = jnp.reshape(res, (b, h, w, c))
        return res



class Unet(hk.Module):
    def __init__(self, dim, groups = 8, dim_scale_factor = (1,2,4,8), name = "unet"):
        super().__init__(name)
        self.name = name
        self.dim = dim
        self.groups = groups
        self.dim_scale_factor = dim_scale_factor
        self.dims = [dim * i for i in dim_scale_factor]

    def __call__(self, x, t):
        channels = x.shape[-1]

        x = hk.Conv2D(self.dim // 3 * 2, 7, padding = ((3,3), (3,3)))(x)
        t_emb = TimeEmbedding(self.dim, "time")(t)

        for index, dim in enumerate(self.dims):
            x = ResnetBlock(dim, self.groups, self.name + "RBd1" + str(index))(x, t_emb)
            x = ResnetBlock(dim, self.groups, self.name + "RBd2" + str(index))(x, t_emb)
            att = ImageMultiHeadAttention(dim, self.groups, "attd" + str(index))(x)
            norm = hk.GroupNorm(self.groups)(att)
            x = x + norm

            if index != len(self.dims) - 1:
                x = hk.Conv2D(dim, 4, 2)(x)

        mid_dim = self.dims[-1]
        x = ResnetBlock(mid_dim, self.groups, self.name + "RBm1")(x, t_emb)
        att = ImageMultiHeadAttention(mid_dim, self.groups, "attm")(x)
        norm = hk.GroupNorm(self.groups)(att)
        x = x + norm
        x = ResnetBlock(mid_dim, self.groups, self.name + "RBm2")(x, t_emb)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = hk.nets.MLP([2048, 10], activation = jax.nn.relu)(x)
        return x




@struct.dataclass
class Params:
    weights: dict
    opt_state: tuple


Unet_model = namedtuple('Nets_and_Opts', 'net optimizer')


def build_model(key, dim, lr):

    net = hk.without_apply_rng(hk.transform(lambda x, t: Unet(dim)(x, t)))
    weights = net.init(key, jnp.zeros((1,dim,dim,1)), jnp.ones((10,), dtype = jnp.int32))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(weights)

    nets_and_opts = Unet_model(net, optimizer)
    params = Params(weights, opt_state)

    return params, nets_and_opts

