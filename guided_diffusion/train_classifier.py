import jax
import jax.numpy as jnp
import optax 


from model_classifier import build_model, Params
from utils import Sampler

def build_trainer(key, dataset, label, batch_size, timesteps, lr = 1e-4):
    n = dataset.shape[0]
    dim = dataset.shape[1]
    n_batch = n//batch_size
    assert n_batch*batch_size == n

    #get model
    params, model = build_model(key, dim, lr)

    #get sample hp
    sampler = Sampler(timesteps)

    def cross_entropy(y, y_pred):
        return jnp.mean(jnp.sum(-y*jax.nn.log_softmax(y_pred), axis = -1))

    def compute_loss(weights, noisy_batch, batch_label, ts):
        pred_label = model.net.apply(weights, noisy_batch, ts)
        true_label = jax.nn.one_hot(batch_label, 10)
        loss = cross_entropy(true_label, pred_label)
        return loss

    compute_grad = jax.value_and_grad(compute_loss)

    
    def batch_iteration(params, batch_indicies_and_key):
        batch_indicies, key = batch_indicies_and_key

        batch = dataset[batch_indicies]
        batch_label = label[batch_indicies]
        ts = jax.random.randint(key, shape = (batch.shape[0],), minval = 0, maxval = timesteps)
        noisy_batch, _ = sampler(key, batch, ts)


        loss, grad = compute_grad(params.weights, noisy_batch, batch_label, ts)


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

def test_build_accuracy(key, net, dataset, label, timesteps,  batch_size = 100, n_epoch = 10):
    n = dataset.shape[0]
    n_batch = n//batch_size
    assert n_batch*batch_size == n

    sampler = Sampler(timesteps)


    def accuracy(y, y_pred):
        y_pred = jax.nn.one_hot(jnp.argmax(y_pred, axis = -1), 10)
        print(y_pred.shape, y.shape)

        return jnp.mean(jnp.sum(y*y_pred, axis = -1))

    def compute_loss(weights, noisy_batch, batch_label, ts):
        pred_label = net.apply(weights, noisy_batch, ts)
        true_label = jax.nn.one_hot(batch_label, 10)
        loss = accuracy(true_label, pred_label)
        return loss
    
    def batch_iteration(weights, batch_indicies_and_key):
        batch_indicies, key = batch_indicies_and_key

        batch = dataset[batch_indicies]
        batch_label = label[batch_indicies]
        ts = jax.random.randint(key, shape = (batch.shape[0],), minval = 0, maxval = timesteps)
        noisy_batch, _ = sampler(key, batch, ts)
        new_loss = compute_loss(weights, noisy_batch, batch_label, ts)
        return weights, new_loss

    def make_epoch(key, tot_loss, weights):
        key, key_step_b, key_model  = jax.random.split(key, 3)
        indicies = jnp.reshape(jax.random.permutation(key_step_b, n), (n_batch, batch_size))
        keys = jax.random.split(key_model, n_batch)
        _, loss = jax.lax.scan(batch_iteration, weights, (indicies, keys))
        
        return key, jnp.sum(loss) + tot_loss

    @jax.jit
    def compute_test_error(key, weights):
        tot_loss = 0
        for _ in range(n_epoch):
            key, tot_loss = make_epoch(key, tot_loss, weights)
        return tot_loss/(n_batch*n_epoch)

    return compute_test_error








from datasets.preprocess import get_dataset
import pickle

train = True

key = jax.random.PRNGKey(0)

dim = 32
epoches = 20
timesteps = 200
x_train, x_test, y_train, y_test = get_dataset(dim)
params, make_epoch, model = build_trainer(key, x_train, y_train, 100, timesteps)
compute_test_error = test_build_accuracy(key, model.net, x_test, y_test, timesteps)

if train:
    for i in range(epoches):
        key, params, loss = make_epoch(key, params)
        test_loss = compute_test_error(key, params.weights)
        print(f"train loss after = {i} steps is {loss}, test loss {test_loss}")
    with open("weights/params_classifier.pkl", "wb") as f:
        pickle.dump(params, f)

else:
    with open("weights/params_classifier.pkl", "rb") as f:
        params = pickle.load(f)





