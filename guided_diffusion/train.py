import jax
import jax.numpy as jnp
import optax 


from model import build_model, Params
from utils import Sampler

def build_trainer(key, dataset, batch_size, timesteps, lr = 1e-4):
    n = dataset.shape[0]
    dim = dataset.shape[1]
    n_batch = n//batch_size
    assert n_batch*batch_size == n

    #get model
    params, model = build_model(key, dim, lr)

    #get sample hp
    sampler = Sampler(timesteps)



    def compute_loss(weights, noisy_batch, noise, ts):
        pred_noise = model.net.apply(weights, noisy_batch, ts)
        loss = jnp.mean(jnp.square(pred_noise - noise))
        return loss

    compute_grad = jax.value_and_grad(compute_loss)

    
    def batch_iteration(params, batch_indicies_and_key):
        batch_indicies, key = batch_indicies_and_key

        batch = dataset[batch_indicies]
        ts = jax.random.randint(key, shape = (batch.shape[0],), minval = 0, maxval = timesteps)
        noisy_batch, noise = sampler(key, batch, ts)


        loss, grad = compute_grad(params.weights, noisy_batch, noise, ts)


        update, opt_state = model.optimizer.update(grad, params.opt_state, params.weights)
        weights = optax.apply_updates(params.weights, update)
        params = Params(weights, opt_state)
        return params, loss

    @jax.jit
    def make_epoch(key, params):
        key, key_step_b, key_model  = jax.random.split(key, 3)
        indicies = jnp.reshape(jax.random.permutation(key_step_b, n), (n_batch, batch_size))
        keys = jax.random.split(key_model, n_batch)
        params, losses = jax.lax.scan(batch_iteration, params, (indicies, keys))
        return key, params, jnp.mean(losses)


    return params, make_epoch, model



