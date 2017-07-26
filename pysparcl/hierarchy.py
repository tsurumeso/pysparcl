from __future__ import print_function

import numpy as np
import six

from pysparcl import internal
from pysparcl import subfunc


def pdist(x, dists=None, wbound=None, metric='squared', niter=15, uorth=None):
    x = x.astype(np.float64)

    if x is None and dists is None:
        raise ValueError('Neither x or dists must not be None.')
    if dists is None:
        n, p = x.shape
        nan_inds = np.isnan(x)
        xnonan = x.copy()
        xnonan[nan_inds] = 0
        dists = internal.distfun(xnonan)
        if np.sum(nan_inds) > 0:
            xbin = np.ones((n, p))
            xbin[nan_inds] = 0
            mult = internal.multfun(xbin)
            if metric == 'squared':
                dists *= np.sqrt(p / np.sum(mult != 0, axis=1))[:, np.newaxis]
            elif metric == 'absolute':
                dists *= (p / np.sum(mult != 0, axis=1))[:, np.newaxis]
            dists[mult == 0] = 0
    if wbound is None:
        wbound = 0.5 * np.sqrt(dists.shape[1])
    if wbound <= 1:
        raise ValueError('wbound must be > 1.')
    if metric == 'squared':
        dists = np.square(dists)
    u, w, crit = subfunc._get_uw(dists, wbound, niter, uorth)
    return u, w, crit, dists


def permute(x, nperms=10, wbounds=None, metric='squared', verbose=False):
    x = x.astype(np.float64)

    if wbounds is None:
        wbounds = np.linspace(1.1, np.sqrt(x.shape[1]) * 0.7, 10)

    tots = np.zeros(len(wbounds))
    permtots = np.zeros((len(wbounds), nperms))
    nnonzerows = np.zeros(len(wbounds))

    u, w, crit, dists = pdist(x, wbound=wbounds[0], metric=metric)
    nnonzerows[0] = np.sum(w != 0)
    tots[0] = crit
    for i in range(1, len(wbounds)):
        result = pdist(None, dists, wbounds[i], metric)
        nnonzerows[i] = np.sum(result[1] != 0)
        tots[i] = result[2]

    permdists = np.zeros(dists.shape)
    for i in range(nperms):
        if verbose:
            six.print_('Permutation %d of %d' % (i + 1, nperms), flush=True)
        for j in range(permdists.shape[1]):
            permdists[:, j] = np.random.permutation(dists[:, j])
        for j in range(len(wbounds)):
            result = pdist(None, permdists, wbounds[j], metric)
            permtots[j, i] = result[2]

    gaps = np.log(tots) - np.log(permtots).mean(axis=1)
    bestw = wbounds[gaps.argmax()]
    return bestw, gaps, wbounds, nnonzerows
