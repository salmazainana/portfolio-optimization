# experiments.py
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import cvxpy as cp

from src.models import SampleAverageApproximation, DistributionallyRobustOptimization

# Synthetic generator 
def simulate_returns(m=10, N=300, seed=None):
    rng = np.random.default_rng(seed)
    psi  = rng.normal(0, 0.02, (N,1))
    means = np.arange(1, m+1) * 0.03
    vars_  = np.arange(1, m+1) * 0.025
    zeta = rng.normal(means, np.sqrt(vars_), (N,m))
    return psi + zeta

# FF10 loader 
def load_ff10(start, end):
    ff10 = web.DataReader('10_Industry_Portfolios', 'famafrench', start, end)[0]
    return ff10.div(100).dropna().values


# # Run SAA vs. DRO with arbitrary loss
# def run_experiment(train_fn, test_fn, epsilons, loss_fn, runs=200):
#     records = []
#     for eps in epsilons:
#         oos_saa_runs = []
#         oos_dro_runs = []
#         for r in range(runs):
#             # regenerate fresh train/test
#             train = train_fn(seed=r)
#             test  = test_fn(seed=r+1000)
#             # solve
#             x_saa, _ = SampleAverageApproximation(train, loss_fn=loss_fn).solve()
#             x_dro, _ = DistributionallyRobustOptimization(train, eps, loss_fn).solve()
#             oos_saa_runs.append(np.mean([-t.dot(x_saa) for t in test]))
#             oos_dro_runs.append(np.mean([-t.dot(x_dro) for t in test]))
#         # average over runs
#         oos_saa = np.mean(oos_saa_runs)
#         oos_dro = np.mean(oos_dro_runs)
#         records.append({'epsilon': eps, 'OOS_SAA': oos_saa, 'OOS_DRO': oos_dro})
#     df = pd.DataFrame(records)
#     df['pct_gain'] = 100*(df['OOS_SAA'] - df['OOS_DRO'])/np.abs(df['OOS_SAA'])
#     return df

def mean_risk_loss(x, xi):
    return -xi.T @ x

def shortfall_loss(x, xi):
    return cp.pos(- xi.T @ x)

# Run SAA vs. DRO with arbitrary loss
def run_experiment(train, test, epsilons, loss_fn):
    records = []
    for eps in epsilons:
        saa = SampleAverageApproximation(train, loss_fn=loss_fn)
        x_saa, _ = saa.solve()
        dro = DistributionallyRobustOptimization(train, epsilon=eps, loss_fn=loss_fn)
        x_dro, _ = dro.solve()
        oos_saa = np.mean([-r.dot(x_saa) for r in test])
        oos_dro = np.mean([-r.dot(x_dro) for r in test])
        records.append({
            'epsilon': eps,
            'OOS_SAA': oos_saa,
            'OOS_DRO': oos_dro
        })
    df = pd.DataFrame(records)
    df['pct_gain'] = 100*(df['OOS_SAA'] - df['OOS_DRO'])/np.abs(df['OOS_SAA'])
    return df

# Weight convergence 
def weight_convergence(N_list, epsilons, runs=100, m=10):
    results = {}
    for N in N_list:
        avg = np.zeros((len(epsilons), m))
        for i, eps in enumerate(epsilons):
            ws = []
            for r in range(runs):
                data = simulate_returns(m, N, seed=r)
                dro = DistributionallyRobustOptimization(data, epsilon=eps, loss_fn=mean_risk_loss)
                ws.append(dro.solve()[0])
            avg[i] = np.mean(ws, axis=0)
        results[N] = avg
    return results

def plot_weights(results, epsilons):
    for N, avg in results.items():
        plt.figure()
        plt.stackplot(epsilons, avg.T)
        plt.xscale('log')
        plt.title(f'Weight convergence, N={N}')
        plt.xlabel('ε')
        plt.ylabel('Weight')
        plt.show()

if __name__ == '__main__':
    # Experiments V2
    epsilons = np.logspace(-4, -1, 20)
    Ns        = [30, 300, 3000]

    # 1) Weight convergence
    wc = weight_convergence(Ns, epsilons, runs=100)
    plot_weights(wc, epsilons)

    # 2) Performance-gap (mean-risk)
    train_syn = simulate_returns(10, 300, seed=0)
    test_syn  = simulate_returns(10, 10000, seed=1)
    df_syn_mr = run_experiment(train_syn, test_syn, epsilons, loss_fn=mean_risk_loss)
    df_syn_mr.to_csv('synthetic_gap_meanrisk.csv', index=False)

    start, end = datetime.datetime(1963,1,1), datetime.datetime(2022,12,31)
    data_ff    = load_ff10(start, end)
    split      = int(0.8*len(data_ff))
    df_ff_mr   = run_experiment(data_ff[:split], data_ff[split:], epsilons, loss_fn=mean_risk_loss)
    df_ff_mr.to_csv('ff10_gap_meanrisk.csv', index=False)

    # 3) Performance-gap (shortfall)
    df_syn_sf = run_experiment(train_syn, test_syn, epsilons, loss_fn=shortfall_loss)
    df_syn_sf.to_csv('synthetic_gap_shortfall.csv', index=False)
    df_ff_sf  = run_experiment(data_ff[:split], data_ff[split:], epsilons, loss_fn=shortfall_loss)
    df_ff_sf.to_csv('ff10_gap_shortfall.csv', index=False)

    # Summaries
    print("Synthetic mean-risk median gain: ", df_syn_mr['pct_gain'].median(), "%")
    print("FF10 mean-risk median gain:      ", df_ff_mr['pct_gain'].median(), "%")
    print("Synthetic shortfall median gain: ", df_syn_sf['pct_gain'].median(), "%")
    print("FF10 shortfall median gain:      ", df_ff_sf['pct_gain'].median(), "%")
#if __name__ == '__main__':
    # Old Experiments 
    # code deleted 


    # Experiments V1
    # epsilons = np.logspace(-4, -1, 20)
    # Ns        = [30, 300, 3000]

    # # 1) Weight convergence
    # wc = weight_convergence(Ns, epsilons, runs=200)
    # plot_weights(wc, epsilons)

    # # 2) Performance-gap (mean-risk)
    # train_syn = simulate_returns(10, 300, seed=0)
    # test_syn  = simulate_returns(10, 10000, seed=1)
    # df_syn_mr = run_experiment(train_syn, test_syn, epsilons, loss_fn=mean_risk_loss)
    # df_syn_mr.to_csv('synthetic_gap_meanrisk.csv', index=False)

    # start, end = datetime.datetime(1963,1,1), datetime.datetime(2022,12,31)
    # data_ff    = load_ff10(start, end)
    # split      = int(0.8*len(data_ff))
    # df_ff_mr   = run_experiment(data_ff[:split], data_ff[split:], epsilons, loss_fn=mean_risk_loss)
    # df_ff_mr.to_csv('ff10_gap_meanrisk.csv', index=False)

    # # 3) Performance-gap (shortfall)
    # df_syn_sf = run_experiment(train_syn, test_syn, epsilons, loss_fn=shortfall_loss)
    # df_syn_sf.to_csv('synthetic_gap_shortfall.csv', index=False)
    # df_ff_sf  = run_experiment(data_ff[:split], data_ff[split:], epsilons, loss_fn=shortfall_loss)
    # df_ff_sf.to_csv('ff10_gap_shortfall.csv', index=False)

    # # Summaries
    # print("Synthetic mean-risk median gain: ", df_syn_mr['pct_gain'].median(), "%")
    # print("FF10 mean-risk median gain:      ", df_ff_mr['pct_gain'].median(), "%")
    # print("Synthetic shortfall median gain: ", df_syn_sf['pct_gain'].median(), "%")
    # print("FF10 shortfall median gain:      ", df_ff_sf['pct_gain'].median(), "%")


 