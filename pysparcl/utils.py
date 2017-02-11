import numpy as np


def _rbind(x0, x1):
    if x0 is None:
        return x1
    else:
        return np.vstack((x0, x1))


def _cbind(x0, x1):
    if x0 is None:
        return x1
    else:
        return np.hstack((x0, x1))


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
