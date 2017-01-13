import numpy as np
import multiprocessing
from scipy.spatial.distance import squareform

import core


def _soft_thresholding(x, d):
    return np.sign(x) * np.maximum(0, np.abs(x) - d)


def _binary_search(argu, sumabs):
    l2n_argu = np.linalg.norm(argu)
    if l2n_argu == 0 or np.sum(np.abs(argu / l2n_argu)) <= sumabs:
        return 0
    lam1 = 0
    lam2 = np.max(np.abs(argu)) - 1e-5
    iter = 1
    while iter <= 15 and (lam2 - lam1) > 1e-4:
        su = _soft_thresholding(argu, (lam1 + lam2) / 2.)
        if np.sum(np.abs(su / np.linalg.norm(su))) < sumabs:
            lam2 = (lam1 + lam2) / 2.
        else:
            lam1 = (lam1 + lam2) / 2.
        iter += 1
    return (lam1 + lam2) / 2.


def _get_uw(ds, wbound, niter, uorth=None):
    n = ds.shape[0]
    p = ds.shape[1]
    u = np.random.randn(p)
    w = (np.ones(p) / p) * wbound
    w_old = np.random.randn(p)
    iter = 1
    while iter <= niter and np.sum(np.abs(w_old - w) / np.sum(np.abs(w_old))) > 1e-4:
        if iter > 1:
            u = ds[:, argw >= lam].dot(w[argw >= lam].T)
        if iter == 1:
            u = ds.dot(w.T)
        iter += 1
        u = u / np.linalg.norm(u)
        w_old = w.copy()
        argw = np.maximum(u.dot(ds), 0).T
        lam = _binary_search(argw, wbound)
        w = _soft_thresholding(argw, lam)
        w = w / np.linalg.norm(w)
    u = ds[:, argw >= lam].dot(w[argw >= lam].T) / np.sum(w)
    u = u / np.linalg.norm(u)
    w = w / np.linalg.norm(w)
    crit = np.sum(u * (ds.dot(w.T)))
    u = squareform(u / np.sqrt(2.))
    return u, w, crit


def pdist(x, dists=None, wbound=None, metric='absolute', niter=15):
    if dists is None:
        dists = core.distfun(x)
        if metric == 'squared':
            dists = dists ** 2
    if wbound is None:
        wbound = 0.5 * np.sqrt(dists.shape[1])
    u, w, crit = _get_uw(dists, wbound, niter)
    return u, w, crit, dists


def _argwrapper(args):
    return args[0](*args[1:])


def _pdist_multiprocess(dists, wbound, metric, id):
    print '_pdist_multiprocess() id: %d' % id
    u, w, crit, dists = pdist(
        x=None, dists=dists, wbound=wbound, metric=metric)
    return id, w, crit


def _permute_multiprocess(permdists, wbounds, metric, id):
    permtot = np.zeros(len(wbounds))
    for i in xrange(permdists.shape[1]):
        permdists[:, i] = np.random.permutation(permdists[:, i])
    for i in xrange(len(wbounds)):
        print '_permute_multiprocess() id: %d -> %d' % (id, i)
        u, w, crit, dists = pdist(
            x=None, dists=permdists, wbound=wbounds[i], metric=metric)
        permtot[i] = np.max(crit)
    return id, permtot.T


def permute(x, nperms=10, wbounds=None, metric='absolute', njobs=None):
    if wbounds is None:
        wbounds = np.linspace(1.1, np.sqrt(x.shape[1]) * 0.7, 10)

    cores = multiprocessing.cpu_count()
    if njobs is None:
        njobs = cores if cores <= 10 else 10
    p = multiprocessing.Pool(njobs)

    tots = np.zeros(len(wbounds))
    permtots = np.zeros((len(wbounds), nperms))
    nnonzerows = np.zeros(len(wbounds))

    u, w, crit, dists = pdist(x, wbound=wbounds[0], metric=metric)
    nnonzerows[0] = np.sum(w != 0)
    tots[0] = crit
    results = p.map(_argwrapper, [(_pdist_multiprocess,
                                  dists, wbounds[i + 1], metric, i + 1)
                                  for i in range(len(wbounds) - 1)])

    for i in xrange(len(wbounds) - 1):
        nnonzerows[results[i][0]] = np.sum(results[i][1] != 0)
        tots[results[i][0]] = results[i][2]

    permdists = dists.copy()
    results = p.map(_argwrapper, [(_permute_multiprocess,
                                  permdists, wbounds, metric, i)
                                  for i in range(nperms)])

    for i in xrange(nperms):
        permtots[:, i] = results[i][1]

    gaps = np.log(tots) - np.log(permtots).mean(axis=1)
    return gaps, wbounds, nnonzerows
