from __future__ import print_function

import numpy as np
import multiprocessing
from pysparcl import core
from pysparcl import internal


def pdist(x, dists=None, wbound=None, metric='squared', niter=15):
    if dists is None:
        dists = internal.distfun(x)
        if metric == 'squared':
            dists = dists ** 2
    if wbound is None:
        wbound = 0.5 * np.sqrt(dists.shape[1])
    u, w, crit = core._get_uw(dists, wbound, niter)
    return u, w, crit, dists


def permute(x, nperms=10, wbounds=None, metric='squared', n_jobs=1):
    if wbounds is None:
        wbounds = np.linspace(1.1, np.sqrt(x.shape[1]) * 0.7, 10)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    p = multiprocessing.Pool(n_jobs)

    tots = np.zeros(len(wbounds))
    permtots = np.zeros((len(wbounds), nperms))
    nnonzerows = np.zeros(len(wbounds))

    u, w, crit, dists = pdist(x, wbound=wbounds[0], metric=metric)
    nnonzerows[0] = np.sum(w != 0)
    tots[0] = crit
    results = p.map(_argwrapper, [(_pdist_multiprocess,
                                  dists, wbounds[i + 1], metric, i + 1)
                                  for i in range(len(wbounds) - 1)])

    for i in range(len(wbounds) - 1):
        nnonzerows[results[i][0]] = np.sum(results[i][1] != 0)
        tots[results[i][0]] = results[i][2]

    permdists = dists.copy()
    results = p.map(_argwrapper, [(_permute_multiprocess,
                                  permdists, wbounds, metric, i)
                                  for i in range(nperms)])

    for i in range(nperms):
        permtots[:, i] = results[i][1]

    p.close()
    gaps = np.log(tots) - np.log(permtots).mean(axis=1)
    bestw = wbounds[gaps.argmax()]
    return bestw, gaps, wbounds, nnonzerows


def _argwrapper(args):
    return args[0](*args[1:])


def _pdist_multiprocess(dists, wbound, metric, id):
    print('_pdist_multiprocess() id: %d' % id)
    u, w, crit, dists = pdist(None, dists, wbound, metric)
    return id, w, crit


def _permute_multiprocess(permdists, wbounds, metric, id):
    permtot = np.zeros(len(wbounds))
    for i in range(permdists.shape[1]):
        permdists[:, i] = np.random.permutation(permdists[:, i])
    for i in range(len(wbounds)):
        print('_permute_multiprocess() id: %d -> %d' % (id, i))
        u, w, crit, dists = pdist(None, permdists, wbounds[i], metric)
        permtot[i] = np.max(crit)
    return id, permtot.T
