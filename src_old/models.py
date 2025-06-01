import numpy as np
import cvxpy as cp
from mosek.fusion import Model, Domain, Expr, ObjectiveSense


class SampleAverageApproximation:
    """Sample Average Approximation (SAA) portfolio optimizer."""
    def __init__(self, returns: np.ndarray, loss_fn=None, solver='ECOS'):
        self.returns = returns
        self.N, self.d = returns.shape
        self.loss_fn = loss_fn if loss_fn is not None else (lambda x, xi: -xi.T @ x)
        self.solver = solver

    def solve(self):
        x = cp.Variable(self.d)
        losses = [self.loss_fn(x, self.returns[i]) for i in range(self.N)]
        avg_loss = cp.sum(losses) / self.N
        problem = cp.Problem(cp.Minimize(avg_loss), [x >= 0, cp.sum(x) == 1])
        problem.solve(solver=self.solver)
        return x.value, problem.value


def simulate_returns(m: int = 10, N: int = 300, seed: int = None) -> np.ndarray:
    """
    Génère des rendements synthétiques indépendants
    avec moyenne mu_i = 0.03 * i
    et écart-type sigma_i = sqrt(0.02^2 + (0.025 * i)^2).
    """
    rng = np.random.default_rng(seed)
    means = np.arange(1, m + 1) * 0.03
    sds = np.sqrt(0.02**2 + (np.arange(1, m + 1) * 0.025)**2)
    return rng.normal(loc=means, scale=sds, size=(N, m))


class WassersteinDRO:
    """
    Distributionally Robust Optimization via Mosek Fusion,
    sous contrainte de simplex et boule de Wasserstein.
    """
    def __init__(self, returns: np.ndarray, epsilon: float):
        self.data = returns
        self.N, self.d = returns.shape
        self.eps = epsilon

    def solve(self):
        with Model("Wasserstein_DRO") as M:
            data_p = M.parameter("data", [self.N, self.d])
            eps_p = M.parameter("eps", 1)

            w = M.variable("w", self.d, Domain.greaterThan(0.0))
            lam = M.variable("lam", 1, Domain.unbounded())
            tau = M.variable("tau", 1, Domain.unbounded())
            slack = M.variable("slack", self.N, Domain.unbounded())

            avg_slack = Expr.sum(slack) / float(self.N)
            M.objective(ObjectiveSense.Minimize,
                        Expr.add(Expr.mul(eps_p, lam), avg_slack))

            M.constraint("sum_to_one", Expr.sum(w), Domain.equalsTo(1.0))

            data_dot = Expr.mul(data_p, w)
            M.constraint("slack_pos",
                         Expr.sub(Expr.repeat(tau, self.N, 1), data_dot),
                         Domain.lessThan(slack))
            M.constraint("slack_neg",
                         Expr.add(Expr.repeat(tau, self.N, 1), data_dot),
                         Domain.lessThan(slack))

            for j in range(self.d):
                M.constraint(f"lam_ge_w_{j}",
                             Expr.sub(lam, Expr.index(w, j)),
                             Domain.greaterThan(0.0))
                M.constraint(f"lam_ge_nw_{j}",
                             Expr.sub(lam, Expr.add(Expr.neg(Expr.index(w, j)), lam)),
                             Domain.greaterThan(0.0))

            data_p.setValue(self.data)
            eps_p.setValue([self.eps])
            M.setSolverParam("optimizer", "freeSimplex")
            M.solve()

            return w.level(), M.primalObjValue()


# class DistributionallyRobustOptimization:
    """Wasserstein DRO with L2-penalization."""
    def __init__(self, returns: np.ndarray, epsilon: float, loss_fn=None, solver='ECOS'):
        self.returns = returns
        self.N, self.d = returns.shape
        self.epsilon = epsilon
        # same default as SAA
        self.loss_fn = loss_fn if loss_fn is not None else (lambda x, xi: -xi.T @ x)
        self.solver = solver

    def solve(self):
        x = cp.Variable(self.d)
        # build average loss
        losses = [ self.loss_fn(x, self.returns[i]) for i in range(self.N) ]
        avg_loss = cp.sum(losses) / self.N
        # L2‐penalty on x
        penalty = self.epsilon * cp.norm(x, 2)
        objective = cp.Minimize(avg_loss + penalty)
        constraints = [x >= 0, cp.sum(x) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)
        return x.value, problem.value

# if __name__ == "__main__":
#     # Quick tests
#     N, d = 100, 4
#     rng = np.random.default_rng(0)
#     returns = rng.normal(size=(N, d))
#     saa = SampleAverageApproximation(returns)
#     print("SAA J_SAA:", saa.solve()[1])
#     # Example piecewise-affine loss: negative return approximated by a max-affine
#     def loss(x, xi):
#         return -xi.T @ x
#     dro = DistributionallyRobustOptimization(returns, epsilon=0.1, loss_fn=loss)
#     print("DRO J_DRO:", dro.solve()[1])

