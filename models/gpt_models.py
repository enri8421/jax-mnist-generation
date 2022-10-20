import jax.numpy as jnp
import jax
import haiku as hk
import optax 
from flax import struct
from collections import namedtuple


class CasualSelfAttention(hk.MultiHeadAttention):
    def __call__(self, x):
        seq_len = x.shape[1]
        casual_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        casual_mask = jnp.tile(casual_mask, (x.shape[0], self.num_heads, 1, 1))
        return super().__call__(x, x, x, casual_mask)


class DecoderBlock(hk.Module):
    def __init__(self, num_heads, hidden_dim, model_size, weight_init_scale, dropout_rate, name):
        super().__init__(name)
        self.casual_atten = CasualSelfAttention(num_heads, hidden_dim, w_init_scale = weight_init_scale, model_size = model_size)
        self.dropout_rate = dropout_rate

    def __call__(self, x, is_training):
        res = self.casual_atten(x)
        if is_training:
            res = hk.dropout(hk.next_rng_key(), self.dropout_rate, res)
        x += res
        x = hk.LayerNorm(-1, True, True)(x)
        dim = x.shape[-1]
        res = hk.Linear(dim * 4)(x)
        res = jax.nn.gelu(res)
        res = hk.Linear(dim)(res)
        if is_training:
            res = hk.dropout(hk.next_rng_key(), self.dropout_rate, res)
        x += res
        x = hk.LayerNorm(-1, True, True)(x)
        return x

class GPTLmHeadModel(hk.Module):
    def __init__(self, num_heads, hidden_dim, num_layers, num_classes, dropout_rate, seq_len, name):
        super().__init__(name)
        model_size = num_heads * hidden_dim
        init_scale = 2.0 / num_layers

        self.embed = hk.Embed(num_classes, model_size)
        self.positional_embeddings = hk.get_parameter("pos_embs", [seq_len, model_size], init=hk.initializers.TruncatedNormal(stddev=0.02))
        self.blocks = [DecoderBlock(num_heads, hidden_dim, model_size, init_scale, dropout_rate, "resblock" + str(i)) for i in range(num_layers)]
        self.lm_head = hk.Linear(num_classes)

    def __call__(self, x, is_training: bool):
        x = self.embed(x) + self.positional_embeddings
        for block in self.blocks:
            x = block(x, is_training)
        x = self.lm_head(x)
        # softmax is taken outside
        return x


@struct.dataclass
class Params:
    weights: dict
    opt_state: tuple


GPT_model = namedtuple('Nets_and_Opts', 'net optimizer')


def build_model(key, num_heads, hidden_dim, num_layers, num_classes, input_size, dropout_rate, lr, weight_decay):

    net = hk.transform(lambda x, is_training: GPTLmHeadModel(num_heads, hidden_dim, num_layers, num_classes, dropout_rate, input_size, "gpt")(x, is_training))
    weights = net.init(key, jnp.zeros((1,input_size), dtype=jnp.int32), True)

    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(weights)

    nets_and_opts = GPT_model(net, optimizer)
    params = Params(weights, opt_state)

    return params, nets_and_opts