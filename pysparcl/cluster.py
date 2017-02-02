from __future__ import print_function

import numpy as np
from pysparcl import core
from pysparcl import utils
from sklearn.cluster import KMeans


def kmeans(x, k=None, wbounds=None, n_init=20, max_iter=6, centers=None, verbose=False):
    n = x.shape[0]
    p = x.shape[1]
    if k is None and centers is None:
        return None
    if not k is None and not centers is None:
        if centers.shape[0] != k or centers.shape[1] != p:
            return None
    if wbounds is None:
        wbounds = np.linspace(1.1, np.sqrt(p), 20)
    if wbounds.min() <= 1:
        return None

    if not centers is None:
        cs = KMeans(k, centers, max_iter=10, n_init=1, n_jobs=1).fit(x).labels_
    else:
        cs = KMeans(k, max_iter=10, n_init=n_init, n_jobs=1).fit(x).labels_

    out = []
    for i in range(len(wbounds)):
        ws = np.ones(p) * (1 / np.sqrt(p))
        ws_old = np.random.standard_normal(p)
        bcss_ws = None
        niter = 0
        while np.sum(np.abs(ws - ws_old)) / np.sum(np.abs(ws_old)) > 1e-4 and niter < max_iter:
            niter += 1
            ws_old = ws
            if niter > 1:
                cs = core._update_cs(x, k, ws, cs)
            ws = core._update_ws(x, cs, wbounds[i])
            bcss_ws = np.sum(core._get_wcss(x, cs)[1] * ws)
        out.append([ws, cs, bcss_ws, wbounds[i]])
        if verbose:
            print('*-------------------------------------------------*')
            print('iter:', i + 1)
            print('wbound:', wbounds[i])
            print('number of non-zero weights:', np.count_nonzero(ws))
            print('sum of weights:', np.sum(ws))
    return out
    

def permute(x, k=None, nperms=25, wbounds=None, nvals=10, centers=None, verbose=False):
    n = x.shape[0]
    p = x.shape[1]
    if wbounds is None:
        wbounds = np.exp(np.linspace(np.log(1.2), np.log(np.sqrt(p) * 0.9), nvals))
    if wbounds.min() <= 1 or len(wbounds) < 2:
        return None
    if not k is None and not centers is None:
        if centers.shape[0] != k or centers.shape[1] != p:
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
        bcss = core._get_wcss(x, out[i][1])[1]
        tots = utils._cbind(tots, np.sum(out[i][0] * bcss))
    permtots = np.zeros((len(wbounds), nperms))
    for i in range(nperms):
        perm_out = kmeans(permx[i], k, wbounds, centers=centers, verbose=verbose)
        for j in range(len(perm_out)):
            perm_bcss = core._get_wcss(permx[i], perm_out[j][1])[1]
            permtots[j, i] = np.sum(perm_out[j][0] * perm_bcss)
    gaps = np.log(tots) - np.log(permtots).mean(axis=1)
    return gaps, wbounds, nnonzerows
