import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.metrics import silhouette_score


def main():
    parser = argparse.ArgumentParser(description='Plot the t-SNE embedding the latent space')
    parser.add_argument('--datadir', type=str, default='./', help='path containing the initial_latent_space.npy and the final_latent_space.npy')
    parser.add_argument('--savedir', type=str, default='./tsne', help='path to save the resulting tsne images')

    opt = parser.parse_args()

    os.makedirs(opt.savedir, exist_ok=True)

    perplexities = [5, 30, 50, 70, 100]

    encoded_latents = np.load(os.path.join(opt.datadir, 'encoded_latent.npy'))
    idx = np.random.choice(encoded_latents.shape[0], 1000)
    encoded_latents = encoded_latents[idx]
    random_latents = np.random.normal(size=encoded_latents.shape)
    latents = np.concatenate([encoded_latents, random_latents], axis=0)
    labels = np.int32(np.concatenate([np.zeros(encoded_latents.shape[0]), np.ones(random_latents.shape[0])]))

    for perplexity in perplexities:
        plot_tsne(latents, labels, perplexity, opt.savedir, 'encoded_latent_perp{}'.format(perplexity))


def plot_tsne(latent, label, perplexity, savedir, savename):
    n_samples = label.shape[0]
    n_components = label.max()+1

    X, y = latent, label.squeeze()

    encoded = y == 0
    random = y == 1

    fig = plt.figure(figsize=(18, 18))
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    plt.scatter(Y[encoded, 0], Y[encoded, 1], c="r", s=400, linewidth=1, edgecolors='black')
    plt.scatter(Y[random, 0], Y[random, 1], c="g", s=400, linewidth=1, edgecolors='black')
    plt.axis('tight')
    plt.savefig(os.path.join(savedir, savename))
    plt.clf()
    plt.close()

if __name__=='__main__':
    main()
