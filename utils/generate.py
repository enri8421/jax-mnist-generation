import jax
import jax.numpy as jnp
import numpy as np

def build_generator(n, decoder_input_size, gtp_weights, vqvae_params, gtp, decoder, embed):
    tokens_size = decoder_input_size[0]*decoder_input_size[1]
    empty_token = jnp.zeros((n*n,tokens_size + 1), dtype = jnp.int32)

    def iteration(i, tokens_and_key):
        tokens, key = tokens_and_key
        y_pred = gtp.apply(gtp_weights, None, tokens, False)
        logits = y_pred[:, i, 10:]/0.5

        key, key_step = jax.random.split(key)
        new_token = jax.random.categorical(key_step, logits)
        tokens = tokens.at[:, i + 1].set(new_token)
        return tokens, key

    def get_tokens(key, label):
        tokens = empty_token.at[:,0].set(label)
        tokens, key = jax.lax.fori_loop(0, tokens_size, iteration, (tokens, key) )
        return tokens, key

    @jax.jit
    def generate(key, label):
        token, _ = get_tokens(key, label)

        z = jnp.reshape(token[:,1:], (n*n,) + decoder_input_size)

        q, _ = embed(vqvae_params.weights, vqvae_params.state, None, z)

        out, _ = decoder(vqvae_params.weights, vqvae_params.state, None, q, False)

        return out

    
    def build_image(seed, label, size):

        out = generate(jax.random.PRNGKey(seed), label)

        image = jnp.zeros((n*size,n*size))
        for i in range(n):
            for j in range(n):
                image = image.at[i*size:(i+1)*size, j*size:(j+1)*size].set(out[i + j*n, :, :, 0])
        return np.array(image*255, dtype = np.uint8)

    return build_image