import cvxpy as cp
import numpy as np

class SampleAverageApproximation:
    """A standard Sample Average Approximation (SAA) portfolio optimization problem:
    min_{x in X} (1/N) * sum_{i=1}^N h(x, ξ_i)
    where h(x, ξ) = -returns(x, ξ) or another loss function of interest.
    This approach is used as a benchmark and to compute the SAA certificate J_SAA 
    as a convex program using CVXPY.
    """
    def __init__(
        self,
        returns: np.ndarray,
        loss_fn: callable = None,
        constraint_fn: callable = None,
        solver: str = 'ECOS'
    ):
        """
        Args:
            returns: An (N, d) NumPy array of scenario returns ξ_i.
            loss_fn: A callable h(x, ξ) returning a CVXPY expression. Defaults to -ξᵀ x.
            constraint_fn: A callable taking decision variable x and returning a list of CVXPY constraints.
                           Defaults to the simplex constraints x >= 0, sum(x) = 1.
            solver: Solver name string for CVXPY (e.g. 'ECOS', 'MOSEK').
        """
        # Data
        self.returns = returns
        self.N, self.d = returns.shape

        # Loss function: default negative return
        if loss_fn is None:
            def default_loss(x, xi):
                return -xi.T @ x
            self.loss_fn = default_loss
        else:
            self.loss_fn = loss_fn

        # Constraint generator: default simplex
        if constraint_fn is None:
            def default_constraints(x):
                return [x >= 0, cp.sum(x) == 1]
            self.constraint_fn = default_constraints
        else:
            self.constraint_fn = constraint_fn

        self.solver = solver

    def solve(self):
        """
        Formulates and solves the SAA problem:
            minimize (1/N) * sum_{i=1}^N h(x, ξ_i)
        Returns:
            x_opt: Optimal decision vector (d,)
            opt_val: Optimal objective value
        """
        # Decision variable
        x = cp.Variable(self.d)

        # Build loss terms for each scenario
        losses = [self.loss_fn(x, self.returns[i]) for i in range(self.N)]
        avg_loss = cp.sum(losses) / self.N

        # Objective and constraints
        objective = cp.Minimize(avg_loss)
        constraints = self.constraint_fn(x)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=getattr(cp, self.solver) if hasattr(cp, self.solver) else None)

        # Store and return results
        self.x_opt = x.value
        self.opt_val = problem.value
        return self.x_opt, self.opt_val
    

class DistributionallyRobustOptimization:
    """A distributionally robust optimization (DRO) problem using the convex reformulation from Section 4.1:
    
        min_{x in X} sup_{Q: d_w(Q, P_N) ≤ ε} E_Q[h(x, ξ)] equivalent to finite convex program per theorem 4.2.
    
    where d_w is the 1-Wasserstein distance, P_N is the empirical distribution and ε the ambiguity radius.
    
    Losses are defined as:
        1.Mean-risk portfolios (piecewise affine loss) → Section 5.1
        2.Uncertainty quantification (indicator/polyhedral loss) → Section 5.2
    """
    def __init__(
        self,
        returns: np.ndarray,
        epsilon: float,
        loss_fn: callable,
        constraint_fn: callable = None,
        solver: str = 'MOSEK'
    ):
        """
        Args:
            returns: An (N, d) NumPy array of scenario returns ξ_i.
            epsilon: Ambiguity radius for Wasserstein ball.
            loss_fn: Callable h(x, ξ) returning CVXPY expression (piecewise or indicator losses).
            constraint_fn: Callable generating CVXPY constraints on x (defaults to simplex).
            solver: CVXPY solver name (e.g., 'MOSEK', 'ECOS').
        """
        self.returns = returns
        self.N, self.d = returns.shape
        self.epsilon = epsilon
        self.loss_fn = loss_fn
        if constraint_fn is None:
            def default_constraints(x):
                return [x >= 0, cp.sum(x) == 1]
            self.constraint_fn = default_constraints
        else:
            self.constraint_fn = constraint_fn
        self.solver = solver

    def solve(self):
        x = cp.Variable(self.d)
        # Auxiliary variable t for the worst-case objective
        t = cp.Variable()
        # Lipschitz constant L of h wrt ξ must be provided or computed
        # Here we assume L known or unity for demonstration
        L = 1.0
        # Reformulation: t >= h(x, ξ_i) for all i, plus epsilon*L term
        constraints = self.constraint_fn(x)
        constraints += [t >= self.loss_fn(x, self.returns[i]) for i in range(self.N)]
        # DRO objective: minimize t + ε * L
        objective = cp.Minimize(t + self.epsilon * L)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=getattr(cp, self.solver) if hasattr(cp, self.solver) else None)
        self.x_opt = x.value
        self.opt_val = problem.value
        return self.x_opt, self.opt_val

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

