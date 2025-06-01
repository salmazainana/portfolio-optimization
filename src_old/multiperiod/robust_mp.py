"""
robust_mp.py – Wasserstein-DRO mean-variance multi-period optimizer.
"""

import cvxpy as cp
import numpy as np

def solve_multiperiod(R: np.ndarray, eps: float = 1e-2, gamma: float = 1.0):
    """
    R     : (T, m) array of returns
    eps   : Wasserstein radius
    gamma : variance penalty weight
    returns: (x_star, info_dict)
    """
    T, m = R.shape
    mu = R.mean(axis=0)
    Sigma = np.cov(R, rowvar=False)

    x = cp.Variable(m, nonneg=True)
    obj = -mu @ x \
          + gamma * cp.quad_form(x, Sigma) \
          + eps * cp.norm(x, 2)
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(x) == 1])

    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS, verbose=False)

    return x.value, {
        'status': prob.status,
        'obj':    prob.value,
        'mu':     mu,
        'Sigma':  Sigma,
        'eps':    eps,
        'gamma':  gamma
    }
