import numpy as np
import matplotlib.pyplot as plt
from DRO import DistributionallyRobustPortfolio, normal_returns

from scipy.special import erfinv


# Experiment parameters
m = 10
N = 300
num_sim = 200
# construct epsilon range: 1e-3 to 1e-1 logarithmically, plus 1.0
epsilons = np.append(
    np.concatenate([np.arange(1, 10) * 10.0**i for i in range(-4, 0)]),
    1.0
)

# Generate independent datasets
datasets = [normal_returns(m, N, seed=i) for i in range(num_sim)]

# Instantiate DRO solver
dro_solver = DistributionallyRobustPortfolio(m, N)

# Storage
weights_accum = np.zeros((len(epsilons), m))
performance = np.zeros((len(epsilons), num_sim))
reliability = np.zeros((len(epsilons), num_sim))

# Known mean and var for analytic performance
mu = np.arange(1, m+1) * 0.03
var = 0.02 + (np.arange(1, m+1) * 0.025)
beta, rho = 0.8, 10
# c2_beta = 1/(np.sqrt(2*np.pi)*(np.exp(np.erfinv(2*beta-1))**2)*(1-beta))
c2_beta = 1/(np.sqrt(2*np.pi)*(np.exp(erfinv(2*beta-1))**2)*(1-beta))

# Run simulations
for j, eps in enumerate(epsilons):
    for i, data in enumerate(datasets):
        w, cert = dro_solver.solve(data, eps)
        weights_accum[j] += w
        # analytic out-of-sample performance
        mean_loss = -w.dot(mu)
        sd_loss = np.sqrt((w**2 * var).sum())
        cVaR = mean_loss + sd_loss * c2_beta
        out_perf = mean_loss + rho * cVaR
        performance[j, i] = out_perf
        reliability[j, i] = float(out_perf <= cert)
    weights_accum[j] /= num_sim

# Compute percentiles for performance
perf_mu = performance.mean(axis=1)
perf_20 = np.quantile(performance, 0.2, axis=1)
perf_80 = np.quantile(performance, 0.8, axis=1)
rel = reliability.mean(axis=1)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

# Weights vs epsilon
for k in range(m):
    lower = weights_accum[:, :k].sum(axis=1) if k>0 else 0
    upper = weights_accum[:, :k+1].sum(axis=1)
    ax[0].fill_between(epsilons, lower, upper,
                       color=plt.cm.tab10(k))
ax[0].set_xscale('log')
ax[0].set_xlabel('epsilon')
ax[0].set_ylabel('Mean portfolio weight')
ax[0].set_title(f'N = {N}')

# Performance & reliability
ax[1].plot(epsilons, perf_mu, color='blue')
ax[1].fill_between(epsilons, perf_20, perf_80, color='blue', alpha=0.3)
ax[1].set_xscale('log')
ax[1].set_xlabel('epsilon')
ax[1].set_ylabel('Out-of-sample performance', color='blue')
ax[1].tick_params(axis='y', labelcolor='blue')

ax2 = ax[1].twinx()
ax2.plot(epsilons, rel, color='red')
ax2.set_ylabel('Reliability', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.tight_layout()
plt.show()