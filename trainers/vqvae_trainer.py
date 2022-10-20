import jax
import jax.numpy as jnp
import optax 


from models.vqvae_models import build_model, Params


def build_trainer(key, dataset, batch_size, latent_dim, num_embeddings, compression_level = 3, res_layers = 2, lr = 3e-4, weight_decay=1e-5, beta = 0.25):
    n = dataset.shape[0]
    input_shape = dataset.shape[1:]
    n_batch = n//batch_size
    assert n_batch*batch_size == n

    params, nets_and_opts = build_model(key, input_shape, latent_dim, num_embeddings, compression_level, res_layers, lr, weight_decay, beta)

    def forward(weights, state, input):
        z_e, state = nets_and_opts.encode(weights, state, None, input, True)
        (loss, q, _), state = nets_and_opts.quantize(weights, state, None, z_e)
        x_pred, state = nets_and_opts.decode(weights, state, None, q, True)
        return x_pred, loss, state


    def compute_loss(weights, state, input):
        output, loss_codebook, state = forward(weights, state, input)
        loss = jnp.mean(jnp.square(input - output)) + loss_codebook
        return loss, state

    compute_grad = jax.value_and_grad(compute_loss, has_aux=True)

    
    def batch_iteration(params, batch_indicies):
        batch = dataset[batch_indicies]
        (loss, state), grad = compute_grad(params.weights, params.state, batch)
        update, opt_state = nets_and_opts.optimizer.update(grad, params.opt_state, params.weights)
        weights = optax.apply_updates(params.weights, update)
        params = Params(weights, opt_state, state)
        return params, loss

    @jax.jit
    def make_epoch(key, params):
        key, key_step = jax.random.split(key)
        indicies = jnp.reshape(jax.random.permutation(key_step, n), (n_batch, batch_size))
        params, losses = jax.lax.scan(batch_iteration, params, indicies)
        return key, params, jnp.mean(losses)

    return params, make_epoch, nets_and_opts