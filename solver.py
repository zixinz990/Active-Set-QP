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

        output_header = '%10s %20s %20s %20s %20s' % \
                       ('iter', 'f_val', 'feas', 'alpha', '||p_k||')

        alpha = 0
        pk = np.zeros(np.shape(xk))
        lambda_sol = np.zeros((np.size(np.where(Wk == True)), 1))
        while True:
            # Evaluate the objective function
            fk = 0.5 * xk.T @ G @ xk + c.T @ xk
            
            # Evaluate the feasibility
            cons_k = A @ xk - b
            feasibility = 0.0
            tmp_1 = np.where(cons_k < 0)[0]
            if np.size(tmp_1) > 0:
                tmp_2 = np.hstack((0, cons_k[tmp_1[0]]))
                feasibility = np.linalg.norm(tmp_2, np.inf)
            
            # Evaluate the first-order optimality
            first_order_opt = np.linalg.norm(G @ xk + c, np.inf)

            # Get the active constraints indices
            active_cons_idx = np.where(Wk == True)[0] + 1
            
            # Print the iteration information
            if self.solver_options.print_level == "Verbose":
                # Print header every 10 iterstion
                if num_iter % 10 == 0:
                    print(output_header)
                output_line = '%10d %20E %20E %20E %20E' % \
                              (num_iter, fk[0][0], feasibility, alpha, np.linalg.norm(pk, 2))
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
                alpha = 0
                # If the KKT conditions are satisfied, return
                if np.all(lambda_sol >= -self.solver_options.term_tol):
                    status = 0
                    if self.solver_options.print_level == "Verbose":
                        fk = 0.5 * xk.T @ G @ xk + c.T @ xk
                        active_cons_idx = np.where(Wk == True)[0] + 1
                        output_line = '%10d %20E %20E %20E %20E' % \
                                      (num_iter+1, fk[0][0], feasibility, alpha, np.linalg.norm(pk, 2))
                        print(output_line)
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
                tmp = np.inf
                for i in range(len(Wk)):
                    if Wk[i] == False and np.dot(A[i, :], pk) < 0:
                        tmp = min(tmp, (b[i] - np.dot(A[i, :], xk)) / np.dot(A[i, :], pk))                       
                alpha = min(1, tmp)
                if alpha != 1:
                    alpha = alpha[0]
                
                # Update xk
                xk = xk + alpha * pk

                # Check if there are blocking constraints
                for i in range(len(Wk)):
                    if Wk[i] == False and abs(np.dot(A[i, :], xk) - b[i]) <= 1e-6:
                        Wk[i] = True
                        break
            
            # Check if the maximum number of iterations is reached
            num_iter += 1
            if num_iter >= self.solver_options.max_iter:
                status = 1
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


np.random.seed(0)
n = 10
m = 10
L = np.random.rand(n,n)
G = np.matmul(L,L.T)
c = 10*(0.5-np.random.rand(n,1))
A = 10*(0.5-np.random.rand(m,n))
b = 10*(0.5-np.random.rand(m,1))
x_feas = np.random.rand(n,1)
b = A @ x_feas - 1
# print("G: ", G)
# print("c: ", c)
# print("A: ", A)
# print("b: ", b)
# print("x_feas: ", x_feas)

x0 = x_feas
W0 = np.array([False]*m) # empty

solver_options = solverOptions()
solver_options.max_iter = 20
solver = Solver(solver_options)
x_sol, lambda_sol, W_sol, status = solver.active_set_solve(G, c, A, b, x0, W0)
print(A @ x_sol - b)