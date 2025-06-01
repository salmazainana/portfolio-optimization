import numpy as np
from scipy.special import erfinv
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from mosek.fusion import Model, Domain, Expr, ObjectiveSense

def normal_returns(m, N, seed=None):
    """
    Generate an N×m matrix of Gaussian returns.
    """
    rng = np.random.default_rng(seed)
    # common noise
    psi   = rng.normal(0, 0.02, size=(N, 1))
    # asset-specific means and variances
    means = np.arange(1, m+1) * 0.03
    vars_ = 0.02 + np.arange(1, m+1) * 0.025
    # idiosyncratic shocks
    zeta  = rng.normal(means, np.sqrt(vars_), size=(N, m))
    return psi + zeta

class DistributionallyRobustPortfolio:
    """
    Base class: builds and solves the single-period Wasserstein-DRO CVaR model via MOSEK Fusion.
    """
    def __init__(self, m, N):
        self.m, self.N = m, N
        self.M = self._build_model(m, N)
        # extract variables & parameters
        self.x   = self.M.getVariable('Weights')
        self.t   = self.M.getVariable('Tau')
        self.dat = self.M.getParameter('TrainData')
        self.eps = self.M.getParameter('WasRadius')
        self.sol_time = []

    def _build_model(self, m, N):
        M = Model(f'DRO_m{m}_N{N}')
        # ----- parameters -----
        dat = M.parameter('TrainData', [N, m])
        eps = M.parameter('WasRadius')
        # CVaR constants for K=2 affine pieces
        a_k = [-1.0, -51.0]
        b_k = [10.0, -40.0]
        # ----- variables -----
        x = M.variable('Weights',     m, Domain.greaterThan(0.0))
        s = M.variable('s_i',         N)
        lam = M.variable('Lambda')
        tau = M.variable('Tau')
        # ----- objective: ε·λ + (1/N)∑s_i -----
        cert = Expr.add(Expr.mul(eps, lam),
                        Expr.mul(Expr.sum(s), 1.0/N))
        M.objective('obj', ObjectiveSense.Minimize, cert)
        # ----- constraints -----
        # C1: b_k·τ + a_k·(x·ξ_i) ≤ s_i for i=1..N, k=1,2
        bk_tau = Expr.hstack([Expr.mul(b, tau) for b in b_k])   # shape 1×2
        e1     = Expr.repeat(bk_tau, N, 0)                      # shape N×2
        dx     = Expr.mul(dat, x)                               # shape N
        e2     = Expr.hstack([Expr.mul(a_k[i], dx) for i in range(2)])  # N×2
        lhs    = Expr.add(e1, e2)
        rhs    = Expr.repeat(s,    2, 1)                        # N×2
        M.constraint('C1', Expr.sub(lhs, rhs), Domain.lessThan(0.0))
        # C2: ||a_k·x||∞ ≤ λ for k=1,2
        e3 = Expr.hstack([Expr.mul(a, x) for a in a_k])        # 2×m
        e4 = Expr.repeat(Expr.repeat(lam, m, 0), 2, 1)          # 2×m
        M.constraint('C2_pos', Expr.sub(e4, e3), Domain.greaterThan(0.0))
        M.constraint('C2_neg', Expr.add(e4, e3), Domain.greaterThan(0.0))
        # C3: simplex ∑x=1
        M.constraint('C3', Expr.sum(x), Domain.equalsTo(1.0))
        M.setSolverParam('optimizer', 'freeSimplex')
        return M

    def sample_average(self, x, tau, data):
        """
        Compute the sample-average CVaR loss for given (x, tau) on 'data'.
        """
        losses = np.maximum(-data.dot(x) + 10*tau,
                             -51*data.dot(x) - 40*tau)
        return losses.mean()

    def iter_data(self, data_list):
        for d in data_list:
            yield self.simulate(d)

    def iter_radius(self, eps_range):
        for eps in eps_range:
            yield self.solve(eps)

    def solve(self, epsilon):
        """
        Solve the DRO model at radius ε on previously set 'TrainData'.
        Returns: (out_of_sample_perf_on_test_data, x, tau, certificate_value)
        """
        self.eps.setValue(epsilon)
        self.M.solve()
        self.sol_time.append(self.M.getSolverDoubleInfo('optimizerTime'))
        x_opt = self.x.level()
        t_opt = self.t.level()
        cert  = self.M.primalObjValue()
        # self.test must be set by subclass before solve()
        oos = self.sample_average(x_opt, t_opt, self.test)
        return oos, x_opt, t_opt, cert

class HoldoutDRO(DistributionallyRobustPortfolio):
    """
    Uses the holdout method: for each dataset, split off 1/k for test,
    tune ε on the train split, then evaluate on a large validation set.
    """
    def __init__(self, m, N, k=5):
        # model sees only (k-1)/k fraction
        n_train = int(N*(k-1)/k)
        super().__init__(m, n_train)
        self.k         = k
        self.eps_range = np.concatenate([np.arange(1,10)*10**i for i in range(-3,0)])
        # large validation set (2e5 samples)
        self.valids    = normal_returns(m, 2*10**5)

    def validate(self, data_list):
        perfs, certs, radii = zip(*self.iter_data(data_list))
        self.perf   = np.stack(perfs,  axis=0)  # shape (200, len(eps_range))
        self.cert   = np.stack(certs, axis=0)
        self.radius = np.mean(radii, axis=0)
        self.rel    = (self.perf <= self.cert).mean(axis=0)

    def simulate(self, data):
        # split train/test
        train, self.test = train_test_split(data, test_size=1/self.k)
        self.dat.setValue(train)
        # sweep radii on the train split
        oos_tr, xs, ts, Js = zip(*self.iter_radius(self.eps_range))
        i_best = int(np.argmin(oos_tr))
        ε_best = self.eps_range[i_best]
        # evaluate performance on the large validation set
        x_best, t_best = xs[i_best], ts[i_best]
        oos_val = self.sample_average(x_best, t_best, self.valids)
        cert    = Js[i_best]
        return oos_val, cert, ε_best

class KFoldDRO(HoldoutDRO):
    """
    k-Fold CV: average the holdout-chosen ε over k random splits, then
    solve on the full dataset of size N.
    """
    def __init__(self, m, N, k=5):
        super().__init__(m, N, k)
        # full-data model (size N)
        self.M_full   = self._build_model(m, N)
        self.dat_full = self.M_full.getParameter('TrainData')
        self.eps_full = self.M_full.getParameter('WasRadius')
        self.x_full   = self.M_full.getVariable('Weights')
        self.t_full   = self.M_full.getVariable('Tau')

    def simulate(self, data):
        # collect k holdout-chosen radii
        eps_chosen = [self._single_holdout(data) for _ in range(self.k)]
        ε_mean = float(np.mean(eps_chosen))
        # solve full-data model at ε_mean
        self.dat_full.setValue(data)
        self.eps_full.setValue(ε_mean)
        self.M_full.solve()
        x_opt = self.x_full.level()
        t_opt = self.t_full.level()
        cert  = self.M_full.primalObjValue()
        # evaluate on the large validation set
        oos_val = self.sample_average(x_opt, t_opt, self.valids)
        return oos_val, cert, ε_mean

    def _single_holdout(self, data):
        # one holdout run for this data
        train, self.test = train_test_split(data, test_size=1/self.k)
        self.dat.setValue(train)
        oos_tr, *_ = zip(*self.iter_radius(self.eps_range))
        return float(self.eps_range[np.argmin(oos_tr)])


class ReliabilityDRO(DistributionallyRobustPortfolio):
    """
    Simulation 3: pick epsilon so that the portfolio is reliable
    at level 1-beta across k random resamples.
    """
    m = 10
    # grid of radii
    eps_range = np.concatenate([np.arange(1, 10)*10.0**i for i in range(-3, 0)])
    # large validation set
    valids    = normal_returns(m, 2 * 10**5)

    def __init__(self, beta, N, k=50):
        self.k    = k          # number of resamples
        self.beta = beta       # 1 - reliability threshold
        super().__init__(ReliabilityDRO.m, N)

    def bootstrap(self, data_list):
        """
        For each dataset in data_list, run simulate() once,
        then collect (perf, cert, chosen_eps) across 200 runs.
        """
        perf, cert, radii = zip(*self.iter_data(data_list))
        # stack into arrays of shape (200, len(eps_range))
        self.perf  = np.stack(perf, axis=0)
        self.cert  = np.stack(cert, axis=0)
        # average chosen radius and reliability
        self.radii = np.mean(radii, axis=0)
        # reliability = fraction of times perf <= cert
        self.rel   = (self.perf <= self.cert).mean(axis=0)

    def simulate(self, data):
        """
        For one dataset:
        - Do k random resamples of train/test (1/3 holdout),
        - For each, tune epsilon by checking reliability on that test,
        - Pick the smallest epsilon whose reliability ≥ 1-beta,
        - Refit on the full data at that epsilon and evaluate on 'valids'.
        Returns (out_perf, cert, chosen_eps).
        """
        counts = np.zeros(len(ReliabilityDRO.eps_range), dtype=int)

        for _ in range(self.k):
            train, self.test = train_test_split(data, test_size=1/3)
            train = resample(train, n_samples=self.N)
            self.dat.setValue(train)

            # for each eps in the grid, solve() returns True/False if
            # test‐loss ≤ certificate
            flags = list(self.iter_radius(ReliabilityDRO.eps_range))
            counts += np.array(flags, dtype=int)

        # find first index where reliability ≥ 1-beta
        threshold = self.k * (1 - self.beta)
        # idx = next(i for i, c in enumerate(counts) if c >= threshold)
        # eps_star = ReliabilityDRO.eps_range[idx]
        # all counts below threshold?
        valid_idxs = [i for i, c in enumerate(counts) if c >= threshold]
        if valid_idxs:
            idx = valid_idxs[0]
        else:
            # fallback: pick epsilon with maximum count
            idx = int(np.argmax(counts))
        eps_star = ReliabilityDRO.eps_range[idx]
        # refit on full data
        self.dat.setValue(data)
        self.eps.setValue(eps_star)
        self.M.solve()
        x_star = self.x.level()
        t_star = self.t.level()
        out_perf = self.sample_average(x_star, t_star, ReliabilityDRO.valids)
        cert     = self.M.primalObjValue()
        return out_perf, cert, eps_star

    def solve(self, epsilon):
        """
        Called by iter_radius: returns True if test‐loss ≤ certificate.
        """
        self.eps.setValue(epsilon)
        self.M.solve()
        x_opt = self.x.level()
        t_opt = self.t.level()
        test_loss = self.sample_average(x_opt, t_opt, self.test)
        return test_loss <= self.M.primalObjValue()

# DRO v2.1
# import numpy as np
# from mosek.fusion import Model, Expr, Domain, ObjectiveSense


# def normal_returns(m, N, seed=None):
#     """
#     Generate synthetic Gaussian returns for m assets over N periods.
#     Means grow linearly and std dev combines a base and an asset-specific term.
#     """
#     rng = np.random.default_rng(seed)
#     R = np.vstack([
#         rng.normal(loc=i*0.03, scale=np.sqrt(0.02**2 + (i*0.025)**2), size=N)
#         for i in range(1, m+1)
#     ])
#     return R.T


# class DistributionallyRobustPortfolio:
#     """
#     Implements Wasserstein DRO with CVaR-style loss via MOSEK Fusion.
#     """
#     def __init__(self, m, N):
#         self.m, self.N = m, N
#         self.M = self.portfolio_model(m, N)
#         self.x = self.M.getVariable('Weights')
#         self.t = self.M.getVariable('Tau')
#         self.dat = self.M.getParameter('TrainData')
#         self.eps = self.M.getParameter('WasRadius')

#     def portfolio_model(self, m, N):
#         M = Model('DistRobust_m{}_N{}'.format(m, N))
#         # Parameters
#         dat = M.parameter('TrainData', [N, m])
#         eps = M.parameter('WasRadius')
#         a_k = [-1, -51]
#         b_k = [10, -40]
#         # Variables
#         x = M.variable('Weights', m, Domain.greaterThan(0.0))
#         s = M.variable('s_i', N)
#         l = M.variable('Lambda')
#         t = M.variable('Tau')
#         # Objective: epsilon * lambda + average s_i
#         # multiply the parameter eps by the variable l
#         sum_s = Expr.sum(s)
#         eps_times_l = Expr.mul(eps, l)
#         # divide sum_s by N by multiplying by 1.0/N
#         avg_s = Expr.mul(sum_s, 1.0/N)
#         certificate = Expr.add(eps_times_l, avg_s)
#         M.objective('J_N(e)', ObjectiveSense.Minimize, certificate)
#         # Constraints: CVaR affine losses
#         # C1: b_k*t + a_k*<x,xi> <= s
#         # b_k * τ term, shape N×2
#         bk_tau    = Expr.hstack([Expr.mul(b_val, t) for b_val in b_k])
#         e1        = Expr.repeat(bk_tau, N, 0)         # N×2

#         # a_k * (dat·x) term, shape N×2
#         data_dot_x = Expr.mul(dat, x)                 # N‐vector
#         e2        = Expr.hstack([Expr.mul(a_k[i], data_dot_x)
#                                 for i in range(2)])   # N×2

#         # Sum them using Expr.add, then compare to s
#         lhs       = Expr.add(e1, e2)                  # N×2
#         rhs       = Expr.repeat(s, 2, 1)              # N×2
#         diff = Expr.sub(lhs, rhs)
#         M.constraint('C1', diff, Domain.lessThan(0.0))
#         # --- C2: Dual Lipschitz bound ---
#         # e3 = [a_k[0]*x, a_k[1]*x] stacked N/A
#         e3 = Expr.hstack([Expr.mul(a_k[i], x) for i in range(2)])
#         # e4 = λ repeated into shape 2×m then N×2
#         e4 = Expr.repeat(Expr.repeat(l, m, 0), 2, 1)

#         # Enforce e4 >= e3  <=>  e4 - e3 >= 0
#         diff_pos = Expr.sub(e4, e3)
#         M.constraint('C2_pos', diff_pos, Domain.greaterThan(0.0))

#         # Enforce e4 >= -e3  <=>  e4 + e3 >= 0
#         diff_neg = Expr.add(e4, e3)
#         M.constraint('C2_neg', diff_neg, Domain.greaterThan(0.0))


#         # --- C3: Simplex constraint sum(x)==1 ---
#         sum_x = Expr.sum(x)
#         M.constraint('C3', sum_x, Domain.equalsTo(1.0))
#         M.setSolverParam('optimizer', 'freeSimplex')
#         return M

#     def solve(self, train_data, epsilon):
#         """
#         Set data and epsilon, solve the DRO model, return weights and certificate value.
#         """
#         self.dat.setValue(train_data)
#         self.eps.setValue(epsilon)
#         self.M.solve()
#         w = self.x.level()
#         cert = self.M.primalObjValue()
#         return w, cert
    
