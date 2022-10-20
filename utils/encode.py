import jax
import jax.numpy as jnp




def compute_size(sample, params, model):
    z_e, _  = model.encode(params.weights, params.state, None, sample, False)
    (_, _, idx), _ = model.quantize(params.weights, params.state, None, z_e)

    assert idx.shape[1] == idx.shape[2]
    return idx.shape[1]



def encode(dataset, params, model, batch_size = 100):
    n = dataset.shape[0]
    @jax.jit
    def compute_embed(inputs):
        z_e, _  = model.encode(params.weights, params.state, None, inputs, False)
        (_, _, idx), _ = model.quantize(params.weights, params.state, None, z_e)
        return idx

    size = compute_size(dataset[:1], params, model)

    encoded = jax.numpy.zeros((n, size, size), dtype=jnp.int32)

    for i in range(n//batch_size):
        encoded = encoded.at[batch_size*i:batch_size*(i+1)].set(compute_embed(dataset[batch_size*i:batch_size*(i+1)]))

    return encoded
