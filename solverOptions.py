'''
Options for the active-set QP solver
'''
class solverOptions:
    def __init__(self):
        self.print_level = "Silent"
        self.term_tol = 1e-6
        self.feas_tol = 1e-9
        self.max_iter = 100
