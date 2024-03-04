import numpy as np

'''
Options for the active-set QP solver
'''
class solverOptions:
    def __init__(self):
        self.print_level = "Silent"
        self.term_tol = 1e-6
        self.feas_tol = 1e-9
        self.max_iter = 100

'''
Active-set QP solver
'''
class Solver:
    def __init__(self, solver_options):
        self.solver_options = solver_options
    
    def active_set_solve(self, G, c, A, b, x0, W0):
        '''
        This function solves a QP with inequality constraints using the active-set method
        
        A QP is defined as follows:
        min  q(x) = 0.5*x'*G*x + c'*x
        s.t. A*x >= b

        Parameters:
            G:  n x n matrix
            c:  n-dim vector
            A:  m x n matrix
            b:  m-dim vector
            x0: starting point
            W0: initial working set, m-dim boolean vector, and a component is set to true if
                the corresponding constraint is in the active set
        
        Returns:
            x_sol:      solution, n-dim vector
            lambda_sol: corresponding optimal Lagrange multipliers, m-dim vector
            W_sol:      final working set, m-dim boolean vector
            status:     exit status

        Note:
            x0 should be feasible, and W0 should be a subset of the active constraints at x0
        '''
        xk = np.copy(x0)
        Wk = np.copy(Wk)
        num_iter = 0

        while True:
            # Initialize active constraints matrix
            A_active = np.copy(A[Wk, :])
            b_active = np.copy(b[Wk])
            
            # Solve the QP with the current working set
            gk = G @ xk + c
            p0 = np.zeros(np.size(gk))
            p_sol, lambda_sol, status = self.solve_eq_qp(G, gk, A_active, b_active, p0)
            
            # If xk is a minimizer with Wk
            if np.linalg.norm(p_sol, 'inf') < self.solver_options.term_tol:
                # If the KKT conditions are satisfied, return
                if np.all(lambda_sol >= -self.solver_options.term_tol):
                    status = 0
                    return xk, lambda_sol, Wk, status
                # Otherwise, remove the constraints with the most negative lambda
                else:
                    Wk[np.argmin(lambda_sol)] = False
            # If xk is not a minimizer with Wk
            else:
                # Find the step length
                alpha = 1
                for i in range(len(Wk)):
                    if Wk[i] == False and np.dot(A[i, :], p_sol) < 0:
                        tmp = (b[i] - np.dot(A[i, :], xk)) / np.dot(A[i, :], p_sol)
                        alpha = min(alpha, tmp)
                
                # Update xk
                xk = xk + alpha * p_sol

                # Check if there are blocking constraints
                for i in range(len(Wk)):
                    if Wk[i] == False and np.dot(A[i, :], xk) - b[i] < -self.solver_options.feas_tol:
                        Wk[i] = True
                        break
            
            # Check if the maximum number of iterations is reached
            num_iter += 1
            if num_iter >= self.solver_options.max_iter:
                status = 1
                return xk, lambda_sol, Wk, status

    def solve_eq_qp(self, G, c, A, b, x0):
        '''
        This function solves a QP with equality constraints using Schur-complement method
        
        A QP is defined as follows:
        min  q(x) = 0.5*x'*G*x + c'*x
        s.t. A*x = b

        Parameters:
            G:  n x n matrix
            c:  n-dim vector
            A:  m x n matrix
            b:  m-dim vector
            x0: starting point
        
        Returns:
            x_sol:       solution, n-dim vector
            lambda_sol:  corresponding optimal Lagrange multipliers, m-dim vector
            status: exit status
        '''
        n = np.shape(x0)[0]
        G_inv = np.linalg.inv(G)
        tmp_inv = np.linalg.inv((A @ G_inv @ A.T))
        C = G_inv - G_inv @ A.T @ tmp_inv @ A @ G_inv
        E = G_inv @ A.T @ tmp_inv
        F = -tmp_inv
        g = c + G @ x0
        h = A @ x0 - b
        K_inv_top = np.hstack((C, E))
        K_inv_bottom = np.hstack((E.T, F))
        K_inv = np.vstack((K_inv_top, K_inv_bottom))
        gh_vec = np.vstack((g, h))
        sol_vec = K_inv @ gh_vec
        p = -sol_vec[:n, :]
        lambda_sol = sol_vec[n:, :]
        x_sol = x0 + p
        status = 0
        return x_sol, lambda_sol, status

solver_options = solverOptions()
solver = Solver(solver_options)
G = np.array([[2, 0], [0, 2]])
c = np.array([[-2], [-5]])
A = np.array([[1, -2], [1, 0]])
b = np.array([[-2], [0]])
x0 = np.array([[2], [0]])
x_sol, lambda_sol, status = solver.solve_eq_qp(G, c, A, b, x0)












