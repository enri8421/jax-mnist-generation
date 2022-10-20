from matplotlib import pyplot as plt



def plot_vqvae(x, params,  model):
    n = x.shape[0]

    z_e, _  = model.encode(params.weights, params.state, None, x, False)
    (_, q, idx), _ = model.quantize(params.weights, params.state, None, z_e)
    x1, _ = model.decode(params.weights, params.state, None, q, False)


    fig = plt.figure(figsize=(12, 8*n))


    for i in range(n):
        fig.add_subplot(n, 3, 3*i + 1)
        plt.imshow(x[i].squeeze())
        fig.add_subplot(n, 3, 3*i +  2)
        plt.imshow(x1[i].squeeze())
        fig.add_subplot(n, 3, 3*i + 3)
        plt.imshow(idx[i].squeeze())


    plt.savefig('vqvae.png')