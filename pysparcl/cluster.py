import numpy as np
import six
from sklearn.cluster import KMeans

from pysparcl import subfunc
from pysparcl import utils


def kmeans(x, k=None, wbounds=None, n_init=20, max_iter=6, centers=None,
           verbose=False):
    n, p = x.shape
    if k is None and centers is None:
        return None
    if k is not None and centers is not None:
        if centers.shape[0] != k or centers.shape[1] != p:
            return None
    if wbounds is None:
        wbounds = np.linspace(1.1, np.sqrt(p), 20)
    if wbounds.min() <= 1:
        return None

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
        out.append([ws, cs, bcss_ws, wbounds[i]])
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
        return None
    if k is not None and centers is not None:
        if centers.shape[0] != k or centers.shape[1] != p:
            return None
    if wbounds is None:
        wbounds = np.exp(
            np.linspace(np.log(1.2), np.log(np.sqrt(p) * 0.9), nvals))
    if wbounds.min() <= 1 or len(wbounds) < 2:
        return None

    permx = np.zeros((nperms, n, p))
    nnonzerows = None
    for i in range(nperms):
        for j in range(p):
            permx[i, :, j] = np.random.permutation(x[:, j])
    tots = None
    out = kmeans(x, k, wbounds, centers=centers, verbose=verbose)

    for i in range(len(out)):
        nnonzerows = utils._cbind(nnonzerows, np.sum(out[i][0] != 0))
        bcss = subfunc._get_wcss(x, out[i][1])[1]
        tots = utils._cbind(tots, np.sum(out[i][0] * bcss))
    permtots = np.zeros((len(wbounds), nperms))
    for i in range(nperms):
        perm_out = kmeans(
            permx[i], k, wbounds, centers=centers, verbose=verbose)
        for j in range(len(perm_out)):
            perm_bcss = subfunc._get_wcss(permx[i], perm_out[j][1])[1]
            permtots[j, i] = np.sum(perm_out[j][0] * perm_bcss)

    gaps = np.log(tots) - np.log(permtots).mean(axis=1)
    bestw = wbounds[gaps.argmax()]
    return bestw, gaps, wbounds, nnonzerows
