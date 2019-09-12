import numpy as np
import six
from sklearn.cluster import KMeans

from pysparcl import subfunc
from pysparcl import utils


def kmeans(x, k=None, wbounds=None, n_init=20, max_iter=6, centers=None,
           verbose=False):
    n, p = x.shape
    if k is None and centers is None:
        raise ValueError('k and centers are None.')
    if k is not None and centers is not None:
        if centers.shape[0] != k or centers.shape[1] != p:
            raise ValueError('Invalid shape of centers.')
    if wbounds is None:
        wbounds = np.linspace(1.1, np.sqrt(p), 20)
    if wbounds.min() <= 1:
        raise ValueError('Each wbound must be > 1.')

    if centers is not None:
        cs = KMeans(centers.shape[0], init=centers, n_init=1).fit(x).labels_
    else:
        cs = KMeans(k, init='random', n_init=n_init).fit(x).labels_

    out = []
    for i in range(len(wbounds)):
        ws = np.ones(p) * (1 / np.sqrt(p))
        ws_old = np.random.standard_normal(p)
        bcss_ws = None
        niter = 0
        while (np.sum(np.abs(ws - ws_old)) / np.sum(np.abs(ws_old)) > 1e-4 and
               niter < max_iter):
            niter += 1
            ws_old = ws
            if niter > 1:
                if k is not None:
                    cs = subfunc._update_cs(x, k, ws, cs)
                else:
                    cs = subfunc._update_cs(x, centers.shape[0], ws, cs)
            ws = subfunc._update_ws(x, cs, wbounds[i])
            bcss_ws = np.sum(subfunc._get_wcss(x, cs)[1] * ws)
        result = {'ws': ws, 'cs': cs, 'bcss_ws': bcss_ws, 'wbound': wbounds[i]}
        out.append(result)
        if verbose:
            six.print_('*-------------------------------------------------*')
            six.print_('iter:', i + 1)
            six.print_('wbound:', wbounds[i])
            six.print_('number of non-zero weights:', np.count_nonzero(ws))
            six.print_('sum of weights:', np.sum(ws), flush=True)
    return out


def permute(x, k=None, nperms=25, wbounds=None, nvals=10, centers=None,
            verbose=False):
    n, p = x.shape
    if k is None and centers is None:
        raise ValueError('k and centers are None.')
    if k is not None and centers is not None:
        if centers.shape[0] != k or centers.shape[1] != p:
            raise ValueError('Invalid shape of centers.')
    if wbounds is None:
        wbounds = np.exp(
            np.linspace(np.log(1.2), np.log(np.sqrt(p) * 0.9), nvals))
    if wbounds.min() <= 1 or len(wbounds) < 2:
        raise ValueError('len(wbounds) and each wbound must be > 1.')

    permx = np.zeros((nperms, n, p))
    nnonzerows = None
    for i in range(nperms):
        for j in range(p):
            permx[i, :, j] = np.random.permutation(x[:, j])
    tots = None
    out = kmeans(x, k, wbounds, centers=centers, verbose=verbose)

    for i in range(len(out)):
        nnonzerows = utils._cbind(nnonzerows, np.sum(out[i]['ws'] != 0))
        bcss = subfunc._get_wcss(x, out[i]['cs'])[1]
        tots = utils._cbind(tots, np.sum(out[i]['ws'] * bcss))
    permtots = np.zeros((len(wbounds), nperms))
    for i in range(nperms):
        perm_out = kmeans(
            permx[i], k, wbounds, centers=centers, verbose=verbose)
        for j in range(len(perm_out)):
            perm_bcss = subfunc._get_wcss(permx[i], perm_out[j]['cs'])[1]
            permtots[j, i] = np.sum(perm_out[j]['ws'] * perm_bcss)

    gaps = np.log(tots) - np.log(permtots).mean(axis=1)
    bestw = wbounds[gaps.argmax()]
    out = {'bestw': bestw, 'gaps': gaps, 'wbounds': wbounds,
           'nnonzerows': nnonzerows}
    return out
