import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import cvxpy as cp

from src.models import SampleAverageApproximation, DistributionallyRobustOptimization
"""
Run Section 4.1 experiments using pre-defined SAA and DRO classes in models.py ( Section 4.1 and 5.1 and 5.2):
  1) Convergence of DRO weights to the uniform portfolio for synthetic 10-asset markets 
  2) Performance gap comparison between SAA and DRO on synthetic and Fama-French 10 datasets
  3) Loss-function comparison: mean-risk vs. shortfall (UQ) losses
"""
# Synthetic data generation (Esfahani & Kuhn, 2018)
def simulate_returns(m=10, N=300, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.normal(loc=0, scale=0.02, size=(N,1))
    means = np.arange(1, m+1) * 0.03
    vars_ = np.arange(1, m+1) * 0.025
    zeta = rng.normal(loc=means, scale=np.sqrt(vars_), size=(N,m))
    return psi + zeta

# Fama-French 10 Industry returns loader
def load_ff10(start, end):
    ff10 = web.DataReader('10_Industry_Portfolios', 'famafrench', start, end)[0]
    return ff10.div(100).dropna().values

# Basic SAA vs DRO experiment
def run_experiment(returns_train, returns_test, epsilons, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x, xi: -xi.T @ x
    records = []
    for eps in epsilons:
        saa = SampleAverageApproximation(returns_train, loss_fn=loss_fn)
        x_saa, J_saa = saa.solve()
        dro = DistributionallyRobustOptimization(returns_train, epsilon=eps, loss_fn=loss_fn)
        x_dro, J_dro = dro.solve()
        losses_saa = np.array([-r.dot(x_saa) for r in returns_test])
        losses_dro = np.array([-r.dot(x_dro) for r in returns_test])
        records.append({
            'epsilon': eps,
            'J_SAA': J_saa,
            'J_DRO': J_dro,
            'OOS_SAA': losses_saa.mean(),
            'OOS_DRO': losses_dro.mean()
        })
    return pd.DataFrame(records)

# Convergence of DRO weights to uniform as ε grows
def weight_convergence_experiment(N_list, epsilons, runs=200, m=10):
    loss_fn = lambda x, xi: -xi.T @ x
    results = {}
    for N in N_list:
        accum = np.zeros((len(epsilons), m))
        for run in range(runs):
            returns_train = simulate_returns(m=m, N=N, seed=run)
            for i, eps in enumerate(epsilons):
                dro = DistributionallyRobustOptimization(returns_train, epsilon=eps, loss_fn=loss_fn)
                x_dro, _ = dro.solve()
                accum[i] += x_dro
        results[N] = accum / runs
    return results

# Plot stacked weights vs ε for each N
def plot_weight_convergence(results, epsilons):
    for N, avg_w in results.items():
        plt.figure()
        plt.stackplot(epsilons, avg_w.T)
        plt.xscale('log')
        plt.xlabel('Wasserstein radius ε')
        plt.ylabel('Average DRO weight')
        plt.title(f'Weight convergence, N={N}')
        plt.legend([f'Asset {j+1}' for j in range(avg_w.shape[1])], loc='upper right')
        plt.tight_layout()
        plt.show()

# Analyze performance gap and print summary stats
def analyze_performance_gap(df, label):
    df = df.copy()
    df['pct_gain'] = (df['J_SAA'] - df['J_DRO']) / df['J_DRO'] * 100
    median = np.median(df['pct_gain'])
    p10 = np.percentile(df['pct_gain'], 10)
    p90 = np.percentile(df['pct_gain'], 90)
    print(f"{label}: median % gain = {median:.1f}%, 10th–90th = {p10:.1f}%–{p90:.1f}%")
    return df

# Plot % improvement vs ε for both datasets
def plot_performance_gap(df_synth, df_ff, epsilons, title_suffix=''):
    df_synth = analyze_performance_gap(df_synth, 'Synthetic')
    df_ff    = analyze_performance_gap(df_ff,    'Fama-French')
    plt.figure()
    plt.plot(epsilons, df_synth['pct_gain'], label='Synthetic')
    plt.plot(epsilons, df_ff['pct_gain'],    label='Fama-French')
    plt.xscale('log')
    plt.xlabel('Wasserstein radius ε')
    plt.ylabel('Relative improvement (%)')
    plt.title(f'DRO vs SAA performance gap {title_suffix}')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Experiment parameters
    epsilons = np.logspace(-4, -1, 20)
    Ns = [30, 300, 3000]
    m = 10
    reps = 200
    print(f"Running experiments with {reps} runs for each N={Ns} and ε={epsilons}")
    # --- 1) Weight convergence ---
    weight_results = weight_convergence_experiment(Ns, epsilons, runs=reps, m=m)
    plot_weight_convergence(weight_results, epsilons)

    print("Weight convergence experiment completed.")

    print("Running performance-gap experiments...")
    # --- 2) Performance-gap experiment ---
    # Synthetic data
    train_syn = simulate_returns(m=m, N=300, seed=0)
    test_syn  = simulate_returns(m=m, N=10000, seed=1)
    results_syn = run_experiment(train_syn, test_syn, epsilons)
    results_syn.to_csv('synthetic_gap.csv', index=False)
    # Fama-French data
    start = datetime.datetime(1963,1,1)
    end   = datetime.datetime(2022,12,31)
    returns_ff = load_ff10(start, end)
    split = int(0.8 * len(returns_ff))
    train_ff, test_ff = returns_ff[:split], returns_ff[split:]
    results_ff = run_experiment(train_ff, test_ff, epsilons)
    results_ff.to_csv('ff10_gap.csv', index=False)
    plot_performance_gap(results_syn, results_ff, epsilons)
    print("Performance-gap experiment completed.")
    print("Running loss-function comparison...")
    # --- 3) Loss-function comparison ---
    # Define two loss functions
    loss_fns = {
        'mean-risk': lambda x, xi: -xi.T @ x,
        'shortfall': lambda x, xi: cp.maximum(0, -xi.T @ x)
    }
    for name, loss_fn in loss_fns.items():
        # Synthetic
        df_m = run_experiment(train_syn, test_syn, epsilons, loss_fn=loss_fn)
        df_m['pct_gain'] = (df_m['OOS_SAA'] - df_m['OOS_DRO']) / np.abs(df_m['OOS_SAA']) * 100
        df_m.to_csv(f'synthetic_gap_{name}.csv', index=False)
        # Fama-French
        df_f = run_experiment(train_ff, test_ff, epsilons, loss_fn=loss_fn)
        df_f['pct_gain'] = (df_f['OOS_SAA'] - df_f['OOS_DRO']) / np.abs(df_f['OOS_SAA']) * 100
        df_f.to_csv(f'ff10_gap_{name}.csv', index=False)
        # plot comparison
        plot_performance_gap(df_m, df_f, epsilons, title_suffix=f'({name})')

    # Summary across datasets for mean-risk
    summary = pd.DataFrame({
        'Dataset': ['Synthetic', 'FF10'],
        'MedianGain': [results_syn['pct_gain'].median(), results_ff['pct_gain'].median()],
        '10thPct': [results_syn['pct_gain'].quantile(0.1), results_ff['pct_gain'].quantile(0.1)],
        '90thPct': [results_syn['pct_gain'].quantile(0.9), results_ff['pct_gain'].quantile(0.9)]
    })
    summary.to_csv('performance_summary.csv', index=False)
    print(summary)
    print("Loss-function comparison completed.")
    print("All experiments completed.")