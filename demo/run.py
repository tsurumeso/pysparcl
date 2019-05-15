from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pysparcl

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


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


if __name__ == '__main__':
    N = 50
    N_dim = 1000
    N_nonzero = 10
    np.random.seed(seed=1)

    class1 = np.zeros(N_dim)
    class2 = np.zeros(N_dim)
    perm = np.random.permutation(N_dim)[:N_nonzero]
    class1[perm] = 1
    class2[perm] = -1

    data = np.vstack(((np.dot(np.ones((N, 1)), [class1]),
                       np.dot(np.ones((N, 1)), [class2]))))
    data += np.random.randn(*data.shape)

    print('Perform hierarchical clustering...')
    dist = pdist(data, 'sqeuclidean')
    show_dendrogram(dist)

    print('Selecting tuning parameter for sparse hierarchical clustering...')
    perm = pysparcl.hierarchy.permute(data, verbose=True)

    print('Perform sparse hierarchical clustering...')
    result = pysparcl.hierarchy.pdist(data, wbound=perm['bestw'])
    show_dendrogram(result['u'])
