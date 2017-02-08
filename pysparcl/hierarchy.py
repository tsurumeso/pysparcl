from __future__ import print_function

import six
import numpy as np
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


def permute(x, nperms=10, wbounds=None, metric='squared', verbose=False):
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

