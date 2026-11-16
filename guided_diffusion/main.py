import jax
import jax.numpy as jnp
import climage
import numpy as np
import pickle
from matplotlib import pyplot as plt

from train import build_trainer
from datasets.preprocess import get_dataset
from utils import build_generator, plot_image, save_gif


key = jax.random.PRNGKey(0)

dim = 32
train = False
load = True

epoches = 20
timesteps = 200

print("getting dataset")

x_train, x_test, y_train, y_test = get_dataset(dim)

print("dataset done!, prepare model")

params, make_epoch, model = build_trainer(key, x_train, 100, timesteps)

print("model ready, start training")

if train:
    if load:
        with open("weights/params2x.pkl", "rb") as f:
            params = pickle.load(f)

    for i in range(epoches):
        key, params, loss = make_epoch(key, params)
        print(f"train loss after = {i} steps is {loss}")
    with open("weights/params3x.pkl", "wb") as f:
        pickle.dump(params, f)
else:
    with open("weights/params3x.pkl", "rb") as f:
        params = pickle.load(f)


print("finished training, build generator")

n_image = 16

generator = build_generator(n_image, params.weights, model.net, timesteps, dim)

print("start generating")

for digit in range(10):
    print("generating digit ", digit)
    x, l = generator(19, digit)


    plot_image(n_image, dim, x, digit)
    save_gif(n_image, dim, l, digit)
