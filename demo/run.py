from __future__ import print_function
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pysparcl

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


def show_dendrogram(dist):
    main_axes = plt.gca()
    divider = make_axes_locatable(main_axes)
    plt.sca(divider.append_axes("top", 1.5, pad=0))

    link = linkage(dist, method='average')
    dendro = dendrogram(
        link, no_labels=True, link_color_func=lambda x: 'black')

    plt.gca().set_axis_off()
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.sca(main_axes)

    distmat = squareform(dist)
    distmat /= distmat.max()
    indices = np.array(dendro['leaves'])
    distmat = (distmat[:, indices])[indices, :]

    plt.imshow(distmat, cmap='Blues', interpolation='nearest')
    plt.show()


def sparse_hierarchical_clustering(data):
    print('Perform hierarchical clustering...')
    dist = pdist(data, 'sqeuclidean')
    show_dendrogram(dist)

    print('Selecting tuning parameter for sparse hierarchical clustering...')
    perm = pysparcl.hierarchy.permute(data, verbose=True)

    print('Perform sparse hierarchical clustering...')
    result = pysparcl.hierarchy.pdist(data, wbound=perm['bestw'])
    show_dendrogram(result['u'])


def sparse_kmeans_clustering(data):
    print('Perform KMeans clustering...')
    print(KMeans(n_clusters=2).fit(data).labels_)

    print('Selecting tuning parameter for sparse KMeans clustering...')
    perm = pysparcl.cluster.permute(data, k=2, verbose=False)

    print('Perform sparse KMeans clustering...')
    result = pysparcl.cluster.kmeans(data, k=2, wbounds=perm['bestw'])
    print(result[0]['cs'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples_per_class', '-n', type=int, default=50)
    p.add_argument('--feature_dims', '-d', type=int, default=1000)
    p.add_argument('--nonzero_features', '-f', type=int, default=10)
    p.add_argument('--random_seed', '-r', type=int, default=1)
    p.add_argument('--mode', '-m', choices=['hierarchy', 'kmeans'], default='hierarchy')
    args = p.parse_args()

    np.random.seed(seed=args.random_seed)

    class1 = np.zeros(args.feature_dims)
    class2 = np.zeros(args.feature_dims)
    perm = np.random.permutation(args.feature_dims)[:args.nonzero_features]
    class1[perm] = 1
    class2[perm] = -1

    data = np.vstack((
        np.dot(np.ones((args.samples_per_class, 1)), [class1]),
        np.dot(np.ones((args.samples_per_class, 1)), [class2])
    ))
    data += np.random.randn(*data.shape)

    if args.mode == 'hierarchy':
        sparse_hierarchical_clustering(data)
    elif args.mode == 'kmeans':
        sparse_kmeans_clustering(data)


if __name__ == '__main__':
    main()
