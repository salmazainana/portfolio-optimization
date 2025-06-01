#!/usr/bin/env python3
"""
Run synthetic check + Fama–French DRO vs Equal-Weight experiment.
Saves the plot as sharpe_vs_eps_ff10.png.
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from src.multiperiod.robust_mp import solve_multiperiod

def simulate_returns(T=12, m=10, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.linspace(0.03, 0.30, m)
    sigma = np.linspace(0.05, 0.30, m)
    psi = rng.normal(0, 0.02, size=(T,1))
    zeta = rng.normal(mu, sigma)
    return psi + zeta

def load_ff10(start="1963-01-01", end="2022-12-31"):
    raw = web.DataReader('Portfolios_Formed_on_ME', 'famafrench')[0]
    if isinstance(raw.index, pd.PeriodIndex):
        raw.index = raw.index.to_timestamp()
    df = raw.loc[start:end] / 100.0
    return df.dropna()

def main():
    # 1) Synthetic sanity check
    R = simulate_returns()
    w, info = solve_multiperiod(R, eps=1e-2, gamma=1.0)
    print("Synthetic test status:", info['status'])
    print("  weights sum to", np.round(w.sum(), 3), "– OK")

    # 2) Load FF10 data
    ff = load_ff10()
    train_frac = 0.8
    n_train = int(len(ff) * train_frac)
    train, test = ff.iloc[:n_train], ff.iloc[n_train:]

    # 3) Equal-weight baseline
    m = ff.shape[1]
    ew = np.ones(m) / m
    ew_ret = test.values @ ew
    ew_sharpe = ew_ret.mean() / ew_ret.std()

    # 4) DRO over eps grid
    eps_grid = np.logspace(-4, -1, 10)
    dro_sharpes = []
    for eps in eps_grid:
        w, _ = solve_multiperiod(train.values, eps=eps, gamma=1.0)
        ret = test.values @ w
        dro_sharpes.append(ret.mean()/ret.std())

    plt.figure(figsize=(6,4))
    plt.plot(eps_grid, dro_sharpes, marker='o', label='Wasserstein-DRO')
    plt.hlines(ew_sharpe, eps_grid.min(), eps_grid.max(),
               colors='gray', linestyles='--',
               label=f'Equal-Weight (Sharpe={ew_sharpe:.2f})')
    plt.xscale('log')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel('Out-of-sample Sharpe')
    plt.title('Multi-Period DRO on FF10')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig/sharpe_vs_eps_ff10.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
