import jax
import jax.numpy as jnp
import climage
import numpy as np
import pickle
from matplotlib import pyplot as plt




from trainers.vqvae_trainer import build_trainer as build_vqvae_trainer
from trainers.gpt_trainer import build_trainer as build_gpt_trainer
from datasets.preprocess import get_dataset
from utils.plot import plot_vqvae
from utils.encode import encode
from utils.generate import build_generator

key = jax.random.PRNGKey(0)

image_size = 24

x_train, x_test, y_train, y_test = get_dataset(image_size)

latent_dim = 128
num_embeddings = 256
batch_size = 100
num_label_class = len(jnp.unique(y_train))

n_vqvae_epoches = 15
n_gpt_epoches = 15

train_vqvae = False
plot = False
train_gtp = False

vqvae_params, make_vqvae_epoch, vqvae_model = build_vqvae_trainer(key, x_train, batch_size, latent_dim, num_embeddings)


if train_vqvae:
    for i in range(n_vqvae_epoches):
        key, vqvae_params, loss = make_vqvae_epoch(key, vqvae_params)
        print(f"loss after = {i} steps is {loss}")
    with open("weights/vqvae_params.pkl", "wb") as f:
        pickle.dump(vqvae_params, f)
else:
    with open("weights/vqvae_params.pkl", "rb") as f:
        vqvae_params = pickle.load(f)


if plot:
    nt = x_test.shape[0]
    indicies = jax.random.randint(key, (20,), 0, nt)
    plot_vqvae(x_test[indicies], vqvae_params, vqvae_model)




train_encoded = encode(x_train, vqvae_params, vqvae_model)
test_encoded = encode(x_test, vqvae_params, vqvae_model)

encoded_shapes = train_encoded.shape[1:]





gtp_params, make_gtp_epoch, gtp_model = build_gpt_trainer(key, train_encoded, y_train, test_encoded, y_test, batch_size, num_label_class +  num_embeddings)

if train_gtp:
    for i in range(n_gpt_epoches):
        key, gtp_params, loss, test_loss = make_gtp_epoch(key, gtp_params)
        print(f"loss after = {i} steps is: train =  {loss}, test = {test_loss}")
    with open("weights/gpt_params.pkl", "wb") as f:
        pickle.dump(gtp_params, f)
else:
    with open("weights/gpt_params.pkl", "rb") as f:
        gtp_params = pickle.load(f)


build_image = build_generator(3, encoded_shapes, gtp_params.weights, vqvae_params, gtp_model.net, vqvae_model.decode,  vqvae_model.embded)



for i in range(num_label_class):
    image = build_image(90, i, image_size)
    plt.figure(figsize=(25,25))
    plt.imshow(image)
    plt.savefig(f'generated_digits/generate_{i}.png')



