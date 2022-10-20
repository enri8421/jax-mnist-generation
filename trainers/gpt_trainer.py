import jax
import jax.numpy as jnp
import optax 


from models.gpt_models import build_model, Params


def build_trainer(key, dataset, labels, testset, testlable, batch_size, num_classes, 
                num_heads = 4, hidden_dim = 256, num_layers = 8, dropout_rate = 0.1, lr = 5e-5, weight_decay=5e-5):

    n = dataset.shape[0]
    nt = testset.shape[0]
    n_batch = n//batch_size
    n_batch_test = nt//batch_size
    assert n_batch*batch_size == n
    assert n_batch_test*batch_size == nt

    tokens = jnp.concatenate((jnp.expand_dims(labels, -1), jnp.reshape(dataset, (n,-1)) + 10), axis = -1)
    test_tokens = jnp.concatenate((jnp.expand_dims(testlable, -1), jnp.reshape(testset, (nt,-1)) + 10), axis = -1)

    input_size = tokens.shape[-1]

    params, nets_and_opts = build_model(key, num_heads, hidden_dim, num_layers, num_classes, input_size, dropout_rate, lr, weight_decay)

    def cross_entropy(y, y_pred):
        return jnp.mean(jnp.sum(-y*jax.nn.log_softmax(y_pred), axis = -1))
    

    @jax.value_and_grad
    def compute_gradient(weights, input, rng):
        y = jax.nn.one_hot(input, num_classes)
        y_pred = nets_and_opts.net.apply(weights, rng, input, True)
        loss = cross_entropy(y[:, 1:], y_pred[:, :-1])
        return loss

    def compute_loss(weights, input):
        y = jax.nn.one_hot(input, num_classes)
        y_pred = nets_and_opts.net.apply(weights, None, input, False)
        loss = cross_entropy(y[:, 1:], y_pred[:, :-1])
        return loss

    def compute_test_loss(weights):
        tot_loss = 0
        for i in range(n_batch_test):
            batch = test_tokens[i*batch_size:(i+1)*batch_size]
            tot_loss += compute_loss(weights, batch)
        return tot_loss/n_batch_test

    def batch_iteration(params, batch_indicies_and_rng):
        (batch_indicies, rng) = batch_indicies_and_rng
        batch = tokens[batch_indicies]
        loss, grad = compute_gradient(params.weights, batch, rng)
        update, opt_state = nets_and_opts.optimizer.update(grad, params.opt_state, params.weights)
        weights = optax.apply_updates(params.weights, update)
        params = Params(weights, opt_state)
        return params, loss

    @jax.jit
    def make_epoch(key, params):
        key, key_step, key_model = jax.random.split(key, 3)
        keys = jax.random.split(key_model, n_batch)
        indicies = jnp.reshape(jax.random.permutation(key_step, n), (n_batch, batch_size))
        params, losses = jax.lax.scan(batch_iteration, params, (indicies, keys))

        test_loss = compute_test_loss(params.weights)
        return key, params, jnp.mean(losses), test_loss

    return params, make_epoch, nets_and_opts