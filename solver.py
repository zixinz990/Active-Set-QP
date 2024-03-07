import numpy as np

'''
Options for the active-set QP solver
'''
class solverOptions:
    def __init__(self):
        self.print_level = "Verbose"
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
        Wk = np.copy(W0)
        num_iter = 0

        if self.solver_options.print_level == "Verbose":
            output_header = '%6s     %20s %9s    %9s' % \
                            ('iter', 'f', 'Wk', 'alpha')
            print(output_header)

        alpha = 0
        while True:
            fk = 0.5 * xk.T @ G @ xk + c.T @ xk

            active_idx = np.where(Wk == True)[0] + 1
            if self.solver_options.print_level == "Verbose":
                output_line = '%6d     %20s %9s         %4s' % \
                            (num_iter, fk[0][0],  active_idx, alpha)
                print(output_line)

            # Initialize active constraints matrix
            A_active = np.copy(A[Wk, :])
            b_active = np.copy(b[Wk])
            
            # Solve the QP with the current working set
            # gk = G @ xk + c
            # p0 = np.zeros(np.shape(gk)) # starting point
            # p_sol, lambda_sol, status = self.solve_eq_qp(G, gk, A_active, b_active, p0)
            x_sol, lambda_sol, status = self.solve_eq_qp(G, c, A_active, b_active, xk)
            pk = x_sol - xk
            # If xk is a minimizer with Wk
            if np.linalg.norm(pk, np.inf) < self.solver_options.term_tol:
                # If the KKT conditions are satisfied, return
                if np.all(lambda_sol >= -self.solver_options.term_tol):
                    status = 0
                    if self.solver_options.print_level == "Verbose":
                        print('Optimal solution found')
                    return xk, lambda_sol, Wk, status
                # Otherwise, remove the constraints with the most negative lambda
                else:
                    idx = np.argmin(lambda_sol) # note that size(lambda_sol) <= size(Wk)
                    is_true_idx = 0
                    for i in range(len(Wk)):
                        if Wk[i] == True:
                            if is_true_idx == idx:
                                Wk[i] = False
                                break
                            else:
                                is_true_idx += 1

            # If xk is not a minimizer with Wk
            else:
                # Find the step length
                alpha = 1
                for i in range(len(Wk)):
                    if Wk[i] == False and np.dot(A[i, :], pk) < 0:
                        tmp = (b[i] - np.dot(A[i, :], xk)) / np.dot(A[i, :], pk)
                        alpha = min(alpha, tmp[0])
                
                # Update xk
                xk = xk + alpha * pk

                # Check if there are blocking constraints
                for i in range(len(Wk)):
                    if Wk[i] == False and np.dot(A[i, :], xk) - b[i] == 0:
                        Wk[i] = True
                        break
            
            # Check if the maximum number of iterations is reached
            num_iter += 1
            if num_iter >= self.solver_options.max_iter:
                status = 1
                if self.solver_options.print_level == "Verbose":
                    print('Maximum number of iterations reached')
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

# Example 1
# min  q(x) = (x1-1)^2 + (x2-2.5)^2
# s.t. -x1 + 2*x2 = -2
#      x2 = 0
# solver_options = solverOptions()
# solver = Solver(solver_options)
# G = np.array([[2, 0], [0, 2]])
# c = np.array([[-2], [-5]])
# A = np.array([[-1, 2], [0, 1]])
# b = np.array([[-2], [0]])

# x0 = np.array([[2], [0]])
# x_sol, lambda_sol, status = solver.solve_eq_qp(G, c, A, b, x0)
# assert np.allclose(x_sol, np.array([[2], [0]]))

# Example 2
# min  q(x) = (x1-1)^2 + (x2-2.5)^2
# s.t. x1 - 2*x2 >= -2
#      -x1 - 2*x2 >= -6
#      -x1 + 2*x2 >= -2
#      x1 >= 0
#      x2 >= 0
# solver_options = solverOptions()
# solver = Solver(solver_options)
# G = np.array([[2, 0], [0, 2]])
# c = np.array([[-2], [-5]])
# A = np.array([[1, -2], [-1, -2], [-1, 2], [1, 0], [0, 1]])
# b = np.array([[-2], [-6], [-2], [0], [0]])

# x0 = np.array([[2], [0]])

# # W0 = np.array([False, False, False, False, False]) # empty
# # W0 = np.array([False, False, True, False, True]) # 3, 5
# # W0 = np.array([False, False, True, False, False]) # 3
# W0 = np.array([False, False, False, False, True]) # 5

# x_sol, lambda_sol, W_sol, status = solver.active_set_solve(G, c, A, b, x0, W0)
# print("x_sol: ", x_sol.flatten())
# print("lambda_sol: ", lambda_sol.flatten())
# print("W_sol: ", np.where(W_sol == True)[0] + 1)

# Example 3
# min  q(x) = x1^2 + x2^2 - 6*x1 - 4*x2
# s.t. -x1 - x2 >= -3
#      x1 >= 0
#      x2 >= 0
solver_options = solverOptions()
solver = Solver(solver_options)
G = np.array([[2, 0], [0, 2]])
c = np.array([[-6], [-4]])
A = np.array([[-1, -1], [1, 0], [0, 1]])
b = np.array([[-3], [0], [0]])

# x0 = np.array([[0], [0]])
x0 = np.array([[1], [1]])

W0 = np.array([False, False, False]) # empty
# W0 = np.array([False, True, True]) # 2, 3

x_sol, lambda_sol, W_sol, status = solver.active_set_solve(G, c, A, b, x0, W0)
print("x_sol: ", x_sol.flatten())
print("lambda_sol: ", lambda_sol.flatten())

