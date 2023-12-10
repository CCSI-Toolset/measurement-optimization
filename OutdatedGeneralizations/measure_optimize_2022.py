"""
Measurement optimization tool 
@University of Notre Dame
"""
from logging import warning
import numpy as np
import pandas as pd
import pyomo.environ as pyo
#import cvxpy as cp
import warnings
from greybox_generalize import LogDetModel
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

class MeasurementOptimizer:
    def __init__(self, static_Q, dynamic_Q, static_Nt, dynamic_Nt, num_param, error_cov=None, verbose=True):
        """
        Argument
        --------
        :param Q: a list of lists containing Jacobian matrix. It should be an m*n matrix, n is No. of parameters, m is the No. of measurements times No. of timepoints.
            Note: Q should be stacked in a way where time points of one measurement are neighbours. For e.g., [CA(t1), ..., CA(tN), CB(t1), ...]
        :param no_measure: No. of measurement items
        :param no_t: No. of time points. Note: this number should be the same for all measurement items
        :param error_cov: 
            If it is a list of lists of shape (Nm*Nt)*(Nm*Nt), containing the variance-covariance matrix of all measurements
            If it is a list of lists of shape Nm*Nm, it will be duplicated for every timepoint
            If it is an Nm*1 vector, these are variances, and covariances are 0. 
            If None, the default error variance-covariance matrix contains 1 for all variances, and 0 for all covariance
            i.e. a diagonal identity matrix
        :param verbose: if print debug sentences
        """
        #self.static_Q = static_Q
        self.num_static = len(static_Q)

        #self.dynamic_Q = dynamic_Q
        self.num_dynamic = len(dynamic_Q)

        self.total_no_measure = self.num_static + self.num_dynamic

        self.static_Nt = static_Nt
        self.dynamic_Nt = dynamic_Nt

        self.num_param = num_param
        self.verbose = verbose

        self.Q = []
        for i in range(self.num_static):
            self.Q.append(static_Q[i])
        for j in range(self.num_dynamic):
            self.Q.append(dynamic_Q[j])

        # check the shape of every input, make sure they are consistent
        #self.__check(Q, no_measure, no_t, cost, error_cov)

        # build and check PSD of Sigma
        self.__build_sigma(error_cov)

    def __check(self, Q, no_measure, no_t, cost, error_cov):
        """
        check if they are legal inputs
        """
        # check shape of Q
        total_no_measure, no_parameter = np.shape(Q)
        if self.verbose:
            print('Q shape:', total_no_measure, no_parameter)

        self.total_no_measure = total_no_measure
        self.no_param = no_parameter

        assert total_no_measure==no_measure*no_t, "Check Q shape!!!"

        # check shape of costs
        assert len(cost)==no_measure*no_t, "Check costs!!!"

    def __build_sigma(self, error_cov):

        self.Sigma_inv = {}
        
        # if getting None: construct an identify matrix
        if error_cov is None:
            # construct identity matrix
            for i in range(self.num_static):
                for j in range(i, self.num_static):
                    self.Sigma_inv[(i,j)] = self.Sigma_inv[(j,i)] = np.identity(self.static_Nt)

            for i in range(self.num_static):
                for j in range(self.num_static, self.total_no_measure):
                    sigma = np.zeros((self.static_Nt, 1))

                    self.Sigma_inv[(i,j)] = self.Sigma_inv[(j,i)] = sigma 

            for i in range(self.num_static, self.total_no_measure):
                for j in range(self.num_static, self.total_no_measure):
                    if i==j:
                        self.Sigma_inv[(i,j)] = 1 
                    else:
                        self.Sigma_inv[(i,j)] = 0

        
    def fim_computation(self):
        """
        compute a list of FIM. 
        """

        self.fim_collection = []

        for i in range(self.total_no_measure):
            for j in range(self.total_no_measure):
                if i < self.num_static and j < self.num_static:
                    unit = np.asarray(self.Q[i]).T@self.Sigma_inv[(i,j)]@np.asarray(self.Q[j])
                    
                elif i<self.num_static and j>=self.num_static:
                    unit = np.asarray(self.Q[i]).T@self.Sigma_inv[(i,j)]@np.asarray(self.Q[j]).reshape(1,self.num_param)

                elif i>=self.num_static and j<self.num_static:
                    unit = np.asarray(self.Q[i]).reshape(1, self.num_param).T@self.Sigma_inv[(i,j)].T@np.asarray(self.Q[j])

                else:
                    unit = self.Sigma_inv[(i,j)]*np.asarray(self.Q[i]).reshape(1, self.num_param).T@np.asarray(self.Q[j]).reshape(1,self.num_param)

                self.fim_collection.append(unit.tolist())

    def compute_FIM(self, measurement_vector):
        """
        Compute a total FIM given a set of measurements

        :param measurement_vector: a list of the length of all measurements, each element in [0,1]
            0 indicates this measurement is not selected, 1 indicates selected
            Note: Ensure the order of this list is the same as the order of Q, i.e. [CA(t1), ..., CA(tN), CB(t1), ...]
        :return:
        """
        # generate measurement matrix
        measurement_matrix = self.__measure_matrix(measurement_vector)

        # compute FIM
        FIM = np.zeros((self.no_param, self.no_param))

        for m1_rank in range(self.total_no_measure):
            for m2_rank in range(self.total_no_measure):
                Q_m1 = np.matrix(self.Q[m1_rank])
                Q_m2 = np.matrix(self.Q[m2_rank])
                measure_matrix = np.matrix([measurement_matrix[m1_rank,m2_rank]])
                sigma = np.matrix([self.Sigma_inv[m1_rank,m2_rank]])
                FIM_unit = Q_m1.T@measure_matrix@sigma@Q_m2
                FIM += FIM_unit
        # FIM read
        if self.verbose:
            self.__print_FIM(FIM)

        return FIM

    def __measure_matrix(self, measurement_vector):
        """

        :param measurement_vector: a vector of measurement weights solution
        :return: a full measurement matrix, construct the weights for covariances
        """
        # check if measurement vector legal
        assert len(measurement_vector)==self.total_no_measure, "Measurement vector is of wrong shape!!!"

        measurement_matrix = np.zeros((self.total_no_measure, self.total_no_measure))

        for i in range(self.total_no_measure):
            for j in range(self.total_no_measure):
                measurement_matrix[i,j] = min(measurement_vector[i], measurement_vector[j])

        return measurement_matrix

    def __print_FIM(self, FIM):
        """

        :param FIM: FIM matrix
        :return: print result analysis
        """

        det = np.linalg.det(FIM)
        trace = np.trace(FIM)
        eig = np.linalg.eigvals(FIM)
        print('======FIM result======')
        print('FIM:', FIM)
        print('Determinant:', det, '; log_e(det):', np.log(det),  '; log_10(det):', np.log10(det))
        print('Trace:', trace, '; log_e(trace):', np.log(trace), '; log_10(trace):', np.log10(trace))
        print('Min eig:', min(eig), '; log_e(min_eig):', np.log(min(eig)), '; log_10(min_eig):', np.log10(min(eig)))
        print('Cond:', max(eig)/min(eig))

    def continuous_optimization(self, cost_list, mixed_integer=False, obj="A", num_fixed=9, 
                                fix=False, sparse=False, init_cov_y=None, init_fim=None,
                                manual_number=20, budget=100):
        
        """Continuous optimization problem formulation. 

        Parameter
        ---------
        :param cost_list: A list, containing the cost of each timepoint of corresponding measurement
        """

        m = pyo.ConcreteModel()

        # response set
        m.NumRes = pyo.Set(initialize=range(self.total_no_measure))
        # FIM set 
        m.DimFIM = pyo.Set(initialize=range(self.num_param))

        # initialize with identity
        def identity(m,a,b):
            return 1 if a==b else 0 
        def initialize_point(m,a,b):
            return init_cov_y[a][b]
        
        if init_cov_y:
            initialize=initialize_point
        else:
            initialize=identity
        
        if mixed_integer:
            m.cov_y = pyo.Var(m.NumRes, m.NumRes, initialize=initialize, within=pyo.Binary)
        else:
            m.cov_y = pyo.Var(m.NumRes, m.NumRes, initialize=initialize, bounds=(0,1), within=pyo.NonNegativeReals)
        
        if fix:
            m.cov_y.fix()

        def init_fim(m,p,q):
            return init_fim[p,q]
        
        m.TotalFIM = pyo.Var(m.DimFIM, m.DimFIM, initialize=identity)

        # other variables

        ### compute FIM 
        def eval_fim(m, a, b):
            if a >= b: 
                summi = 0 
                for i in m.NumRes:
                    for j in m.NumRes:
                        if i>j:
                            summi += m.cov_y[i,j]*self.fim_collection[i*self.total_no_measure+j][a][b]
                        else:
                            summi += m.cov_y[j,i]*self.fim_collection[i*self.total_no_measure+j][a][b]
                return m.TotalFIM[a,b] == summi
            else:
                return m.TotalFIM[a,b] == m.TotalFIM[b,a]
            
        ### cov_y constraints
        def y_covy1(m, a, b):
            if a > b:
                return m.cov_y[a, b] <= m.cov_y[a, a]
            else:
                return pyo.Constraint.Skip
            
        def y_covy2(m, a, b):
            if a > b:
                return m.cov_y[a, b] <= m.cov_y[b, b]
            else:
                return pyo.Constraint.Skip
            
        def y_covy3(m, a, b):
            if a>b:
                return m.cov_y[a, b] >= m.cov_y[a, a] + m.cov_y[b, b] - 1
            else:
                return pyo.Constraint.Skip
            
        def symmetry(m,a,b):
            if a<b:
                return m.cov_y[a,b] == m.cov_y[b,a]
            else:
                return pyo.Constraint.Skip

        ### cost constraints
        def cost_compute(m):
            return m.cost == sum(m.cov_y[i,i]*cost_list[i] for i in m.NumRes)
        
        def cost_limit(m):
            return m.cost <= budget
        
        def total_dynamic(m):
            return m.TotalDynamic==sum(m.cov_y[i,i] for i in range(num_fixed, self.total_no_measure))
        
        def total_dynamic_con(m):
            return m.TotalDynamic<=manual_number
        
        # set up Design criterion
        def ComputeTrace(m):
            sum_x = sum(m.TotalFIM[j,j] for j in m.DimFIM)
            return sum_x

        ### add constraints
        m.TotalFIM_con = pyo.Constraint(m.DimFIM, m.DimFIM, rule=eval_fim)

        if mixed_integer and not fix:
            m.sym = pyo.Constraint(m.NumRes, m.NumRes, rule=symmetry)
        
        if not fix:
            if not sparse:
                m.cov1 = pyo.Constraint(m.NumRes, m.NumRes, rule=y_covy1)
                m.cov2 = pyo.Constraint(m.NumRes, m.NumRes, rule=y_covy2)
                m.cov3 = pyo.Constraint(m.NumRes, m.NumRes, rule=y_covy3)
                
            else: 
                m.cov1_sparse = pyo.Constraint(m.NumRes_half, rule=y_covy1)
                m.cov2_sparse = pyo.Constraint(m.NumRes_half, rule=y_covy2)
                m.cov3_sparse = pyo.Constraint(m.NumRes_half, rule=y_covy3)
                

            m.TotalDynamic = pyo.Var(initialize=1) # total # of human measurements
            m.con_manual = pyo.Constraint(rule=total_dynamic_con)
            m.cost = pyo.Var(initialize=budget)
            m.cost_compute = pyo.Constraint(rule=cost_compute)
            m.budget_limit = pyo.Constraint(rule=cost_limit)

        # set objective 
        if obj == "A":
            m.Obj = pyo.Objective(rule=ComputeTrace, sense=pyo.maximize)

        elif obj == "D":

            def _model_i(b):
                self.build_model_external(b)
            m.my_block = pyo.Block(rule=_model_i)

            for i in range(self.num_param):
                for j in range(i, self.num_param):
                    def eq_fim(m):
                        return m.TotalFIM[i,j] == m.my_block.egb.inputs["ele_"+str(i)+"_"+str(j)]
                    
                    con_name = "con"+str(i)+str(j)
                    m.add_component(con_name, pyo.Constraint(expr=eq_fim))

            # add objective
            m.Obj = pyo.Objective(expr=m.my_block.egb.outputs['log_det'], sense=pyo.maximize)

        return m 


    def build_model_external(self, m):
        ex_model = LogDetModel(num_para=5)
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)

    def continuous_optimization_cvxpy(self, objective='D', budget=100, solver=None):
        """

        :param objective: can choose from 'D', 'A', 'E' for now. if defined others or None, use A-optimality.
        :param cost_budget: give a total limit for costs.
        :param solver: default to be MOSEK. Look for CVXPY document for more solver information.
        :return:
        """

        # compute Atomic FIM
        self.fim_computation()

        # evaluate fim 
        def eval_fim(y):
            fim = sum(y[i,j]*self.fim_collection[i*self.total_no_measure+j] for i in range(self.total_no_measure) for j in range(self.total_no_measure))
            return fim

        def a_opt(y):
            fim = eval_fim(y)
            return cp.trace(fim)
            
        def d_opt(y):
            fim = eval_fim(y)
            return cp.log_det(fim)

        def e_opt(y):
            fim = eval_fim(y)
            return -cp.lambda_min(fim)

        # construct variables
        y_matrice = cp.Variable((self.total_no_measure,self.total_no_measure), nonneg=True)

        # cost limit 
        p_cons = [sum(y_matrice[i,i]*self.cost[i] for i in range(self.total_no_measure)) <= budget]

        # constraints
        for k in range(self.total_no_measure):
            for l in range(self.total_no_measure):
                p_cons += [y_matrice[k,l] <= y_matrice[k,k]]
                p_cons += [y_matrice[k,l] <= y_matrice[l,l]]
                p_cons += [y_matrice[k,k] + y_matrice[l,l] -1 <= y_matrice[k,l]] 
                p_cons += [y_matrice.T == y_matrice]


        if objective == 'D':
            obj = cp.Maximize(d_opt(y_matrice))
        elif objective =='E':
            obj = cp.Maximize(e_opt(y_matrice))
        else:
            # 
            if self.verbose:
                print("Use A-optimality (Trace).")
            obj = cp.Maximize(a_opt(y_matrice))

        problem = cp.Problem(obj, p_cons)

        if not solver:
            problem.solve(verbose=True)
        else:
            problem.solve(solver=solver, verbose=True)

        self.__solution_analysis(y_matrice, obj.value)
            

    

    def __solution_analysis(self, y_value, obj_value):
        """
        Analyze solution. Rounded solutions, test if they meet constraints, and print information about the solution.

        :param y_value: cvxpy problem output y 
        :param obj_value: cvxopy problem output objective function value
        """

        ## deal with y solution, round
        sol = np.zeros((self.total_no_measure,self.total_no_measure))

        # get all solution value. If a solution is larger than 0.99, round it to 1. if it is smaller than 0.01, floor it to 0.
        for i in range(self.total_no_measure):
            for j in range(self.total_no_measure):
                sol[i,j] = y_value[i,j].value
                
                if sol[i,j] >0.99:
                    sol[i,j] = 1
                    
                if sol[i,j] <0.01:
                    sol[i,j] = 0
        
        ## test solution 
        for i in range(self.total_no_measure):
            for j in range(self.total_no_measure):
                if abs(sol[i,j]-sol[j,i])>0.01:
                    print('Covariance between measurements {i_n} and {j_n} has wrong symmetry'.format(i_n=i, j_n=j))
                    print('Cov 1:' , sol[i,j] , ', Cov 2:' , sol[j,i] + '.')
                    
                if abs(sol[i,j]-min(sol[i,i], sol[j,j]))>0.01:
                    print('Covariance between measurements {i_n} and {j_n} has wrong computation'.format(i_n=i, j_n=j))
                    print('i weight:', sol[i,i] , ', j weight:', sol[j,j] , '; Cov weight:', sol[i,j])

        ## obj

        print("Objective:", obj_value)
        
        solution_choice = []
        
        for i in range(self.total_no_measure):
            solution_choice.append(sol[i,i])

        ## check if obj right
        print('FIM info verification (The following result is computed by compute_FIM method)')
        real_FIM = self.compute_FIM(solution_choice)

        

                


class DataProcess:
    def __init__(self) -> None:
        """
        """

        return 

    def read_jaco(self, filename):

        jaco_info = pd.read_csv(filename, index_col=False)
        jaco_list = np.asarray(jaco_info)

        jaco = []
        for i in range(len(jaco_list)):
            # jacobian remove fisrt column
            jaco.append(list(jaco_list[i][1:]))

        print("jacobian shape:", np.shape(jaco))

        return jaco

    def split_jaco(self, jaco, idx, num_t):
        """Split Jacobian 
        idx: idx of static measurements 
        """
        jaco_idx = jaco[idx*num_t:(idx+1)*num_t][:]
        return jaco_idx 


    

