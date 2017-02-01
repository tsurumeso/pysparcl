import core
import numpy as np
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

    labels = []
    for i in range(len(wbounds)):
        ws = np.ones(p, dtype=np.float32) * (1 / np.sqrt(p))
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
        labels.append(cs)
        if verbose:
            print '*-------------------------------------------------*'
            print 'iter:', i + 1
            print 'wbound:', wbounds[i]
            print 'number of non-zero weights:', np.count_nonzero(ws)
            print 'sum of weights:', np.sum(ws)
            print 'labels:\n', cs
    return labels, wbounds
