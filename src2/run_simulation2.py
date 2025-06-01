import os
import numpy as np
import matplotlib.pyplot as plt
from DRO import normal_returns, HoldoutDRO, KFoldDRO

# locate the image2 folder (one level up from src2)
SRC_DIR   = os.path.dirname(__file__)
IMAGE_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir, "image2"))

# sample‐size grid
N_range = np.append(
    np.concatenate([np.arange(1,10)*10**i for i in range(1,3)]),
    1000
)

def plot_experiment(results, title, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3), dpi=150)
    fig.suptitle(title)

    # 1) Out‐of‐sample performance
    perf = np.stack([r.perf for r in results], axis=0)
    mean_perf = perf.mean(axis=1)
    lo_perf   = np.quantile(perf, 0.2, axis=1)
    hi_perf   = np.quantile(perf, 0.8, axis=1)
    ax = axes[0]
    ax.fill_between(N_range, lo_perf, hi_perf, alpha=0.4)
    ax.plot(N_range, mean_perf, marker='>')
    ax.set(xscale='log', xlabel='N',
           ylabel='Out‐of‐sample performance')
    ax.grid(True)

    # 2) Certificate
    cert = np.stack([r.cert for r in results], axis=0)
    mean_cert = cert.mean(axis=1)
    lo_cert   = np.quantile(cert, 0.2, axis=1)
    hi_cert   = np.quantile(cert, 0.8, axis=1)
    ax = axes[1]
    ax.fill_between(N_range, lo_cert, hi_cert, alpha=0.4)
    ax.plot(N_range, mean_cert, marker='>')
    ax.set(xscale='log', xlabel='N',
           ylabel='Certificate')
    ax.grid(True)

    # 3) Reliability
    rels = np.array([r.rel for r in results])
    ax = axes[2]
    ax.plot(N_range, rels, marker='>')
    ax.set(xscale='log', xlabel='N',
           ylabel='Reliability', ylim=[0,1])
    ax.grid(True)

    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()

def run_holdout():
    results = []
    for N in N_range:
        data_list = [normal_returns(10, int(N), seed=i) for i in range(200)]
        hld = HoldoutDRO(m=10, N=int(N), k=5)
        hld.validate(data_list)
        results.append(hld)
    plot_experiment(results,
                    "Simulation 2: Holdout Method",
                    os.path.join(IMAGE_DIR, "figure3.png"))

def run_kfold():
    results = []
    for N in N_range:
        data_list = [normal_returns(10, int(N), seed=i) for i in range(200)]
        kfld = KFoldDRO(m=10, N=int(N), k=5)
        kfld.validate(data_list)
        results.append(kfld)
    plot_experiment(results,
                    "Simulation 2: 5-Fold Cross-Validation",
                    os.path.join(IMAGE_DIR, "figure4.png"))

if __name__ == "__main__":
    # print("Running holdout experiment...")
    # run_holdout()
    print("Running 5-fold CV experiment...")
    run_kfold()
    print("Done. Figures saved in:", IMAGE_DIR)
