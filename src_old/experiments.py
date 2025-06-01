import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import cvxpy as cp

from src.models import (
    simulate_returns,
    SampleAverageApproximation,
    WassersteinDRO
)

# Loader des portfolios Fama-French 10 industries

def load_ff10(start: datetime.datetime, end: datetime.datetime) -> np.ndarray:
    ff10 = web.DataReader('10_Industry_Portfolios', 'famafrench', start, end)[0]
    return ff10.div(100).dropna().values

# Fonctions de perte

def mean_risk_loss(x, xi):
    return -xi.T @ x

def shortfall_loss(x, xi):
    return cp.pos(-xi.T @ x)

# Expérience SAA vs DRO

def run_experiment(train: np.ndarray,
                   test: np.ndarray,
                   epsilons: np.ndarray,
                   loss_fn) -> pd.DataFrame:
    records = []
    for eps in epsilons:
        # SAA
        saa = SampleAverageApproximation(train, loss_fn=loss_fn)
        w_saa, _ = saa.solve()

        # DRO
        dro = WassersteinDRO(train, epsilon=eps)
        w_dro, _ = dro.solve()

        # Performance out-of-sample
        oos_saa = np.mean([-t.dot(w_saa) for t in test])
        oos_dro = np.mean([-t.dot(w_dro) for t in test])

        records.append({
            'epsilon': eps,
            'OOS_SAA': oos_saa,
            'OOS_DRO': oos_dro
        })

    df = pd.DataFrame(records)
    df['pct_gain'] = 100 * (df['OOS_SAA'] - df['OOS_DRO']) / np.abs(df['OOS_SAA'])
    return df

# Convergence des poids

def weight_convergence(N_list, epsilons, runs=100, m=10):
    results = {}
    for N in N_list:
        avg_weights = []
        for eps in epsilons:
            weights = []
            for r in range(runs):
                data = simulate_returns(m, N, seed=r)
                dro = WassersteinDRO(data, epsilon=eps)
                w, _ = dro.solve()
                weights.append(w)
            avg_weights.append(np.mean(weights, axis=0))
        results[N] = np.array(avg_weights)
    return results

# Visualisation

def plot_weights(results: dict, epsilons: np.ndarray):
    for N, w_avg in results.items():
        plt.figure()
        plt.stackplot(epsilons, w_avg.T)
        plt.xscale('log')
        plt.title(f'Weight Convergence, N={N}')
        plt.xlabel('Epsilon')
        plt.ylabel('Weights')
        plt.show()

if __name__ == '__main__':
    epsilons = np.logspace(-4, -1, 20)
    Ns = [30, 300, 3000]

    # 1) Convergence des poids
    wc = weight_convergence(Ns, epsilons, runs=100)
    plot_weights(wc, epsilons)

    # 2) Écart de performance (mean-risk)
    train_syn = simulate_returns(10, 300, seed=0)
    test_syn = simulate_returns(10, 10000, seed=1)
    df_syn_mr = run_experiment(train_syn, test_syn, epsilons, mean_risk_loss)
    df_syn_mr.to_csv('synthetic_gap_meanrisk.csv', index=False)


    # Fama-French
    start = datetime.datetime(1963, 1, 1)
    end = datetime.datetime(2022, 12, 31)
    data_ff = load_ff10(start, end)
    split = int(0.8 * len(data_ff))
    df_ff_mr = run_experiment(data_ff[:split], data_ff[split:], epsilons, mean_risk_loss)
    df_ff_mr.to_csv('ff10_gap_meanrisk.csv', index=False)



    # 3) Écart de performance (shortfall)
    df_syn_sf = run_experiment(train_syn, test_syn, epsilons, shortfall_loss)
    df_syn_sf.to_csv('synthetic_gap_shortfall.csv', index=False)
    df_ff_sf = run_experiment(data_ff[:split], data_ff[split:], epsilons, shortfall_loss)
    df_ff_sf.to_csv('ff10_gap_shortfall.csv', index=False)

    # --- Impact of the Wasserstein radius ---

    plt.figure()
    plt.plot(epsilons, df_syn_mr['pct_gain'], marker='o', label='Synthétique (mean-risk)')
    plt.plot(epsilons, df_ff_mr['pct_gain'], marker='s', label='Fama–French (mean-risk)')
    plt.plot(epsilons, df_syn_sf['pct_gain'], marker='.', label='Synthétique (short-fall)')
    plt.plot(epsilons, df_ff_sf['pct_gain'], marker='--', label='Fama–French (short-fall)')

    plt.xscale('log')
    plt.xlabel('Wasserstein radius (ε)')
    plt.ylabel('Gain de performance (%)')
    plt.title('Impact de ε sur le gain out‐of‐sample')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('impact_wasserstein_radius.png')
    # Résumés

    print("Synthetic mean-risk median gain: ", df_syn_mr['pct_gain'].median(), "%")
    print("FF10 mean-risk median gain:      ", df_ff_mr['pct_gain'].median(), "%")
    print("Synthetic shortfall median gain: ", df_syn_sf['pct_gain'].median(), "%")
    print("FF10 shortfall median gain:      ", df_ff_sf['pct_gain'].median(), "%")


    # plot pour out of sample performance
    
# import numpy as np
# import pandas as pd
# import datetime
# import pandas_datareader.data as web
# import matplotlib.pyplot as plt
# import cvxpy as cp

# from src.models import SampleAverageApproximation, DistributionallyRobustOptimization

# # Synthetic generator 
# def simulate_returns(m=10, N=300, seed=None):
#     rng = np.random.default_rng(seed)
#     # psi  = rng.normal(0, 0.02, (N,1))
#     means = np.arange(1, m+1) * 0.03
#     sds   = np.sqrt(0.02**2 + (np.arange(1, m+1) * 0.025)**2)
#     R = rng.normal(loc=means, scale=sds, size=(N, m))
#     return R

# # FF10 loader 
# def load_ff10(start, end):
#     ff10 = web.DataReader('10_Industry_Portfolios', 'famafrench', start, end)[0]
#     return ff10.div(100).dropna().values


# # # Run SAA vs. DRO with arbitrary loss
# # def run_experiment(train_fn, test_fn, epsilons, loss_fn, runs=200):
# #     records = []
# #     for eps in epsilons:
# #         oos_saa_runs = []
# #         oos_dro_runs = []
# #         for r in range(runs):
# #             # regenerate fresh train/test
# #             train = train_fn(seed=r)
# #             test  = test_fn(seed=r+1000)
# #             # solve
# #             x_saa, _ = SampleAverageApproximation(train, loss_fn=loss_fn).solve()
# #             x_dro, _ = DistributionallyRobustOptimization(train, eps, loss_fn).solve()
# #             oos_saa_runs.append(np.mean([-t.dot(x_saa) for t in test]))
# #             oos_dro_runs.append(np.mean([-t.dot(x_dro) for t in test]))
# #         # average over runs
# #         oos_saa = np.mean(oos_saa_runs)
# #         oos_dro = np.mean(oos_dro_runs)
# #         records.append({'epsilon': eps, 'OOS_SAA': oos_saa, 'OOS_DRO': oos_dro})
# #     df = pd.DataFrame(records)
# #     df['pct_gain'] = 100*(df['OOS_SAA'] - df['OOS_DRO'])/np.abs(df['OOS_SAA'])
# #     return df

# def mean_risk_loss(x, xi):
#     return -xi.T @ x

# def shortfall_loss(x, xi):
#     return cp.pos(- xi.T @ x)

# # Run SAA vs. DRO with arbitrary loss
# def run_experiment(train, test, epsilons, loss_fn):
#     records = []
#     for eps in epsilons:
#         saa = SampleAverageApproximation(train, loss_fn=loss_fn)
#         x_saa, _ = saa.solve()
#         dro = DistributionallyRobustOptimization(train, epsilon=eps, loss_fn=loss_fn)
#         x_dro, _ = dro.solve()
#         oos_saa = np.mean([-r.dot(x_saa) for r in test])
#         oos_dro = np.mean([-r.dot(x_dro) for r in test])
#         records.append({
#             'epsilon': eps,
#             'OOS_SAA': oos_saa,
#             'OOS_DRO': oos_dro
#         })
#     df = pd.DataFrame(records)
#     df['pct_gain'] = 100*(df['OOS_SAA'] - df['OOS_DRO'])/np.abs(df['OOS_SAA'])
#     return df

# # Weight convergence 
# def weight_convergence(N_list, epsilons, runs=100, m=10):
#     results = {}
#     for N in N_list:
#         avg = np.zeros((len(epsilons), m))
#         for i, eps in enumerate(epsilons):
#             ws = []
#             for r in range(runs):
#                 data = simulate_returns(m, N, seed=r)
#                 dro = DistributionallyRobustOptimization(data, epsilon=eps, loss_fn=mean_risk_loss)
#                 ws.append(dro.solve()[0])
#             avg[i] = np.mean(ws, axis=0)
#         results[N] = avg
#     return results

# def plot_weights(results, epsilons):
#     for N, avg in results.items():
#         plt.figure()
#         plt.stackplot(epsilons, avg.T)
#         plt.xscale('log')
#         plt.title(f'Weight convergence, N={N}')
#         plt.xlabel('ε')
#         plt.ylabel('Weight')
#         plt.show()

# if __name__ == '__main__':
#     # Experiments V2
#     epsilons = np.logspace(-4, -1, 20)
#     Ns        = [30, 300, 3000]

#     # 1) Weight convergence
#     wc = weight_convergence(Ns, epsilons, runs=100)
#     plot_weights(wc, epsilons)

#     # 2) Performance-gap (mean-risk)
#     train_syn = simulate_returns(10, 300, seed=0)
#     test_syn  = simulate_returns(10, 10000, seed=1)
#     df_syn_mr = run_experiment(train_syn, test_syn, epsilons, loss_fn=mean_risk_loss)
#     df_syn_mr.to_csv('synthetic_gap_meanrisk.csv', index=False)

#     start, end = datetime.datetime(1963,1,1), datetime.datetime(2022,12,31)
#     data_ff    = load_ff10(start, end)
#     split      = int(0.8*len(data_ff))
#     df_ff_mr   = run_experiment(data_ff[:split], data_ff[split:], epsilons, loss_fn=mean_risk_loss)
#     df_ff_mr.to_csv('ff10_gap_meanrisk.csv', index=False)

#     # 3) Performance-gap (shortfall)
#     df_syn_sf = run_experiment(train_syn, test_syn, epsilons, loss_fn=shortfall_loss)
#     df_syn_sf.to_csv('synthetic_gap_shortfall.csv', index=False)
#     df_ff_sf  = run_experiment(data_ff[:split], data_ff[split:], epsilons, loss_fn=shortfall_loss)
#     df_ff_sf.to_csv('ff10_gap_shortfall.csv', index=False)

#     # Summaries
#     print("Synthetic mean-risk median gain: ", df_syn_mr['pct_gain'].median(), "%")
#     print("FF10 mean-risk median gain:      ", df_ff_mr['pct_gain'].median(), "%")
#     print("Synthetic shortfall median gain: ", df_syn_sf['pct_gain'].median(), "%")
#     print("FF10 shortfall median gain:      ", df_ff_sf['pct_gain'].median(), "%")
# #if __name__ == '__main__':
#     # Old Experiments 
#     # code deleted 


#     # Experiments V1
#     # epsilons = np.logspace(-4, -1, 20)
#     # Ns        = [30, 300, 3000]

#     # # 1) Weight convergence
#     # wc = weight_convergence(Ns, epsilons, runs=200)
#     # plot_weights(wc, epsilons)

#     # # 2) Performance-gap (mean-risk)
#     # train_syn = simulate_returns(10, 300, seed=0)
#     # test_syn  = simulate_returns(10, 10000, seed=1)
#     # df_syn_mr = run_experiment(train_syn, test_syn, epsilons, loss_fn=mean_risk_loss)
#     # df_syn_mr.to_csv('synthetic_gap_meanrisk.csv', index=False)

#     # start, end = datetime.datetime(1963,1,1), datetime.datetime(2022,12,31)
#     # data_ff    = load_ff10(start, end)
#     # split      = int(0.8*len(data_ff))
#     # df_ff_mr   = run_experiment(data_ff[:split], data_ff[split:], epsilons, loss_fn=mean_risk_loss)
#     # df_ff_mr.to_csv('ff10_gap_meanrisk.csv', index=False)

#     # # 3) Performance-gap (shortfall)
#     # df_syn_sf = run_experiment(train_syn, test_syn, epsilons, loss_fn=shortfall_loss)
#     # df_syn_sf.to_csv('synthetic_gap_shortfall.csv', index=False)
#     # df_ff_sf  = run_experiment(data_ff[:split], data_ff[split:], epsilons, loss_fn=shortfall_loss)
#     # df_ff_sf.to_csv('ff10_gap_shortfall.csv', index=False)

#     # # Summaries
#     # print("Synthetic mean-risk median gain: ", df_syn_mr['pct_gain'].median(), "%")
#     # print("FF10 mean-risk median gain:      ", df_ff_mr['pct_gain'].median(), "%")
#     # print("Synthetic shortfall median gain: ", df_syn_sf['pct_gain'].median(), "%")
#     # print("FF10 shortfall median gain:      ", df_ff_sf['pct_gain'].median(), "%")


 