import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

from pysparcl import utils


def _get_uw(ds, wbound, niter, uorth=None):
    n, p = ds.shape
    u = np.random.randn(p)
    w = (np.ones(p) / p) * wbound
    w_old = np.random.standard_normal(p)
    iter = 0
    if uorth is not None:
        if np.sum(np.abs(uorth - uorth.T)) > 1e-10:
            return None
        uorth = squareform(uorth)
        uorth /= np.sqrt(np.sum(np.square(uorth)))
    while (iter < niter and
           np.sum(np.abs(w_old - w) / np.sum(np.abs(w_old))) > 1e-4):
        if iter == 0:
            u = ds.dot(w.T)
        else:
            u = ds[:, argw >= lam].dot(w[argw >= lam].T)
        if uorth is not None:
            u -= uorth.dot(uorth.T.dot(u))
        iter += 1
        u = u / np.linalg.norm(u)
        w_old = w.copy()
        argw = np.maximum(u.dot(ds), 0).T
        lam = utils._binary_search(argw, wbound)
        w = utils._soft_thresholding(argw, lam)
        w /= np.linalg.norm(w)
    u = ds[:, argw >= lam].dot(w[argw >= lam].T) / np.sum(w)
    if uorth is not None:
        u -= uorth.dot(uorth.T.dot(u))
    u /= np.linalg.norm(u)
    w /= np.linalg.norm(w)
    crit = np.sum(u * (ds.dot(w.T)))
    u /= np.sqrt(2.)
    return u, w, crit


def _get_wcss(x, cs, ws=None):
    wcss_perf = np.zeros(x.shape[1])
    for i in np.unique(cs):
        mask = (cs == i)
        if np.sum(mask) > 1:
            wcss_perf += np.sum(
                np.square(x[mask, :] - np.mean(x[mask, :], axis=0)), axis=0)
    bcss_perf = np.sum(np.square(x - np.mean(x, axis=0)), axis=0) - wcss_perf
    return wcss_perf, bcss_perf


def _update_cs(x, k, ws, cs):
    x = x[:, ws != 0]
    z = x * np.sqrt(ws[ws != 0])
    nrowz = z.shape[0]
    mus = None
    if cs is not None:
        for i in np.unique(cs):
            if np.sum(cs == i) > 1:
                mus = utils._rbind(mus, np.mean(z[cs == i, :], axis=0))
            if np.sum(cs == i) == 1:
                mus = utils._rbind(mus, z[cs == i, :])
    if mus is None:
        km = KMeans(k, init='random', n_init=10).fit(z)
    else:
        distmat = squareform(pdist(utils._rbind(z, mus)))
        distmat = distmat[:nrowz, (nrowz + 1):(nrowz + k)]
        nearest = distmat.argmin(axis=1)
        if len(np.unique(nearest)) == k:
            km = KMeans(k, init=mus, n_init=1).fit(z)
        else:
            km = KMeans(k, init='random', n_init=10).fit(z)
    return km.labels_


def _update_ws(x, cs, wbound):
    wcss_perf = _get_wcss(x, cs)[0]
    tss_perf = _get_wcss(x, np.ones(x.shape[0]))[0]
    lam = utils._binary_search(-wcss_perf + tss_perf, wbound)
    ws_unscaled = utils._soft_thresholding(-wcss_perf + tss_perf, lam)
    return ws_unscaled / np.linalg.norm(ws_unscaled)
