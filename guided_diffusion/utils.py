import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from model_classifier import build_model, Params
import pickle


class Sampler(object):
    def __init__(self, timesteps):
        beta = jnp.linspace(0.0001, 0.02, timesteps)
        alpha = 1 - beta
        alpha_bar = jnp.cumprod(alpha, 0)
        alpha_bar = jnp.concatenate((jnp.array([1.]), alpha_bar[:-1]), axis=0)
        self.sqrt_alpha_bar = jnp.sqrt(alpha_bar)
        self.one_minus_sqrt_alpha_bar = jnp.sqrt(1 - alpha_bar)

    # Implement noising logic according to reparameterization trick
    def __call__(self, key, x_0, t):
        noise = jax.random.normal(key, x_0.shape)  
        reshaped_sqrt_alpha_bar_t = jnp.reshape(jnp.take(self.sqrt_alpha_bar, t), (-1, 1, 1, 1))
        reshaped_one_minus_sqrt_alpha_bar_t = jnp.reshape(jnp.take(self.one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))
        noisy_image = reshaped_sqrt_alpha_bar_t  * x_0 + reshaped_one_minus_sqrt_alpha_bar_t  * noise
        return noisy_image, noise


def build_get_grad(key, dim):
    _, model = build_model(key, dim, 1)
    with open("weights/params_classifier.pkl", "rb") as f:
        params = pickle.load(f)

    def compute_log_prob(image, t, label):
        logits = model.net.apply(params.weights, image, t)
        log_prob = jnp.take(logits, label) - jax.nn.logsumexp(logits)
        return log_prob

    compute_grad_vec = jax.vmap(jax.grad(compute_log_prob), in_axes = (0,0,None))

    @jax.jit
    def get_gradient(image, t, label):
        image = jnp.expand_dims(image, 1)
        t = jnp.ones( (image.shape[0], 1), dtype = jnp.int32)*t.squeeze()
        return jnp.expand_dims(compute_grad_vec(image, t, label).squeeze(), -1)

    return get_gradient

    
def build_generator(n, weights, net, timesteps, size,  inference_timesteps = 4):

    beta = jnp.linspace(0.0001, 0.02, timesteps)
    alpha = 1 - beta
    alpha_bar = jnp.cumprod(alpha, 0)
    alpha_bar = jnp.concatenate((jnp.array([1.]), alpha_bar[:-1]), axis=0)

    inference_range = jnp.flip(jnp.linspace(0, timesteps, num=inference_timesteps, endpoint=False, dtype = jnp.int32))

    get_gradient = build_get_grad(jax.random.PRNGKey(0), size)

    def backward_denoising_ddim(x_t, pred_noise, t, digit):
        guidance = get_gradient(x_t, t, digit)
        alpha_bar_t = jnp.take(alpha_bar, t)
        alpha_t_minus_one = jnp.take(alpha, t - 1)
        pred_noise -= ((1 - alpha_bar_t) ** 0.5)*guidance
        pred = (x_t - ((1 - alpha_bar_t) ** 0.5) * pred_noise)/ (alpha_bar_t ** 0.5)
        pred = (alpha_t_minus_one ** 0.5) * pred
        pred = pred + ((1 - alpha_t_minus_one) ** 0.5) * pred_noise
        return pred

    def loop_iter(digit, x, idx):
        t = jnp.expand_dims(idx, 0) 
        pred_noise = net.apply(weights, x, jnp.ones(n, dtype = jnp.int32)*idx)
        x = backward_denoising_ddim(x, pred_noise, t, digit)
        return x, x.squeeze(axis = -1)    

    @jax.jit
    def generate_image(seed, digit):
        key = jax.random.PRNGKey(seed)
        key, start_key = jax.random.split(key)
        img_list_ddim = jnp.zeros((inference_timesteps + 1, n, size, size))
        x = jax.random.normal(start_key, (n, size, size, 1))
        img_list_ddim = img_list_ddim.at[0].set(x.squeeze(axis = -1))

        x, l = jax.lax.scan(lambda x, idx: loop_iter(digit, x, idx), x, inference_range)
        img_list_ddim = img_list_ddim.at[1:].set(l)

        return x.squeeze(-1), img_list_ddim

    return generate_image


def display_img(images, n_tot, size):
    n = int(np.sqrt(n_tot))
    assert n*n == n_tot
    image = jnp.zeros((n*size,n*size))
    for i in range(n):
        for j in range(n):
            image = image.at[i*size:(i+1)*size, j*size:(j+1)*size].set(images[i + j*n])

    image = np.array( (image + 1)*127.5, dtype = np.uint8)
    return image



def plot_image(n, size, images, digit):
    image = display_img(images, n, size)
    plt.figure(figsize=(25,25))
    plt.imshow(image)
    plt.savefig(f'generated_digits/generate_{digit}.png')

def save_gif(n, size, images, digit):
    imgs = (Image.fromarray(display_img(images[i], n, size)) for i in range(images.shape[0]))
    # Extract first image from iterator
    img = next(imgs)  

    # Append the other images and save as GIF
    img.save(fp=f"generated_digits_process/process_{digit}.gif", format='GIF', append_images=imgs,
             save_all=True, duration=1000, loop=0)