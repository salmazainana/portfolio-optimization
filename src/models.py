import cvxpy as cp
import numpy as np

class SampleAverageApproximation:
    """Sample Average Approximation (SAA) portfolio optimizer."""
    def __init__(self, returns: np.ndarray, loss_fn=None, solver='ECOS'):
        """
        returns: (N × d) array of scenario returns
        loss_fn: function (x, xi) → CVXPY expression; defaults to -xi^T x
        """
        self.returns = returns
        self.N, self.d = returns.shape
        self.loss_fn = loss_fn if loss_fn is not None else (lambda x, xi: -xi.T @ x)
        self.solver = solver

    def solve(self):
        x = cp.Variable(self.d)
        losses = [ self.loss_fn(x, self.returns[i]) for i in range(self.N) ]
        avg_loss = cp.sum(losses) / self.N
        problem = cp.Problem(cp.Minimize(avg_loss),
                             [ x >= 0, cp.sum(x) == 1 ])
        problem.solve(solver=self.solver)
        return x.value, problem.value

class DistributionallyRobustOptimization:
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

