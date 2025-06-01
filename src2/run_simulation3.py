import os
import numpy as np
import matplotlib.pyplot as plt
from DRO import ReliabilityDRO, normal_returns

# locate image2 folder
SRC = os.path.dirname(__file__)
IMAGE_DIR = os.path.abspath(os.path.join(SRC, os.pardir, "image2"))

# sample‐size grid
N_values = np.append(
    np.concatenate([np.arange(1,10)*10**i for i in range(1,3)]),
    1000
)

def N_plot(results, title, outpath, y_l):
    fig, axes = plt.subplots(1,3,figsize=(13,3),dpi=150)
    fig.suptitle(title)

    # out‐of‐sample performance
    perf = np.stack([r.perf for r in results], axis=0)
    m_perf = perf.mean(axis=1)
    lo_perf = np.quantile(perf,0.2,axis=1)
    hi_perf = np.quantile(perf,0.8,axis=1)
    ax = axes[0]
    ax.fill_between(N_values, lo_perf, hi_perf, alpha=0.4)
    ax.plot(N_values, m_perf, marker='>')
    ax.set(xscale='log',xlabel='N',ylabel='Out‐of‐sample performance',ylim=y_l[0])
    ax.grid(True)

    # certificate
    cert = np.stack([r.cert for r in results], axis=0)
    m_cert = cert.mean(axis=1)
    lo_cert = np.quantile(cert,0.2,axis=1)
    hi_cert = np.quantile(cert,0.8,axis=1)
    ax = axes[1]
    ax.fill_between(N_values, lo_cert, hi_cert, alpha=0.4)
    ax.plot(N_values, m_cert, marker='>')
    ax.set(xscale='log',xlabel='N',ylabel='Certificate',ylim=y_l[1])
    ax.grid(True)

    # reliability
    rel = np.array([r.rel for r in results])
    ax = axes[2]
    ax.plot(N_values, rel, marker='>')
    ax.set(xscale='log',xlabel='N',ylabel='Reliability',ylim=y_l[2])
    ax.grid(True)

    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.show()

if __name__ == '__main__':
    beta = 0.1
    results = []
    for N in N_values:
        datasets = [normal_returns(10, int(N), seed=i) for i in range(200)]
        runner   = ReliabilityDRO(beta, int(N), k=50)
        runner.bootstrap(datasets)
        results.append(runner)

    outpath = os.path.join(IMAGE_DIR, 'figure5.png')
    N_plot(
        results,
        f"Simulation 3: Reliability ≥ {1-beta:.1f}",
        outpath,
        y_l=[[-1.5,1.5], [-2.5,10], [0,1]]
    )
    print("Saved reliability plot to", outpath)
