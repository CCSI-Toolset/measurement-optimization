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



class DataProcess:
    def __init__(self) -> None:
        return 
    
    def read_jaco(self, filename):
        """Read Jacobian from csv file 
        """
        jaco_info = pd.read_csv(filename, index_col=False)
        jaco_list = np.asarray(jaco_info)

        jaco = []
        for i in range(len(jaco_list)):
            # jacobian remove fisrt column
            jaco.append(list(jaco_list[i][1:]))

        print("jacobian shape:", np.shape(jaco))
        return jaco

    def split_jaco(self, jaco, idx, num_t):
        """Split Jacobian according to measurements
        idx: idx of static measurements 
        """
        jaco_idx = jaco[idx*num_t:(idx+1)*num_t][:]
        return jaco_idx 

class MeasurementOptimizer:
    def __init__(self, Q, static_idx, dynamic_idx, num_param, error_cov=None, error_opt=None, verbose=True):
        """
        Argument
        --------
        :param Q: a list of lists containing Jacobian matrix. 
            It contains m lists, m is the No. of meausrements 
            Each list contains an N_t_m*num_param elements, which is the sensitivity matrix Q for measurement m 
        :param static_idx: a list of static-cost measurements index 
        :param dynamic_idx: a list of dynamic-cost measurements index 
        :param num_param: No. of parameters
        :param error_cov: 
            If it is a list of lists of shape (Nm*Nt)*(Nm*Nt), containing the variance-covariance matrix of all measurements
            If it is a list of lists of shape Nm*Nm, it will be duplicated for every timepoint
            If it is an Nm*1 vector, these are variances, and covariances are 0. 
            If None, the default error variance-covariance matrix contains 1 for all variances, and 0 for all covariance
            i.e. a diagonal identity matrix
        :param: error_opt;
            1: variances given
            2: 
        :param verbose: if print debug sentences
        """
        # # of static and dynamic measurements
        self.num_static = len(static_idx)
        self.static_idx = static_idx
        self.num_dynamic = len(dynamic_idx)
        self.dynamic_idx = dynamic_idx
        self.num_measure = len(Q)
        assert self.num_measure==self.num_dynamic+self.num_static

        # measurements can have different # of timepoints
        self.Nt = {}
        for i in range(self.num_measure):
            self.Nt[i] = len(Q[i])
        # total number of measurement and time points
        self.total_num_time = sum(self.Nt.values())

        self.num_param = num_param
        self.verbose = verbose

        # flattened Q and indexes
        self._dynamic_flatten(Q)

        # check the shape of every input, make sure they are consistent
        # TO BE ADDED after deciding on user interface
        #self.__check(Q, no_measure, no_t, cost, error_cov)

        # build and check PSD of Sigma
        Sigma = self._build_sigma(error_cov, error_opt)
        self._split_sigma(Sigma)

    def _dynamic_flatten(self, Q):
        """Update dynamic flattened matrix index. 
        dynamic_flatten matrix: flatten dynamic-cost measurements, not flatten static-costs, [s1, d1|t1, ..., d1|tN, s2]
        Flatten matrix: flatten dynamic-cost and static-cost measuremenets
        """

        ### flatten to be cov matrix 
        Q_dynamic_flatten = []
        self.head_pos_dynamic_flatten = {}
        self.static_idx_dynamic_flatten = []
        self.dynamic_idx_dynamic_flatten = []

        Q_flatten = []
        self.head_pos_flatten = {}
        self.static_idx_flatten = []
        self.dynamic_idx_flatten = []

        self.dynamic_to_flatten = {}

        count1 = 0
        count2 = 0
        for i in range(self.num_measure):
            if i in self.static_idx:
                # dynamic_flatten
                Q_dynamic_flatten.append(Q[i])
                #print(np.shape(Q[i]))
                self.head_pos_dynamic_flatten[i] = count1 
                self.static_idx_dynamic_flatten.append(count1)
                self.dynamic_to_flatten[count1] = []

                # flatten 
                for t in range(len(Q[i])):
                    Q_flatten.append(Q[i][t])
                    if t==0:
                        self.head_pos_flatten[i] = count2
                    self.static_idx_flatten.append(count2)
                    # map
                    self.dynamic_to_flatten[count1].append(count2)
                    count2 += 1 

                count1 += 1 

            else:
                for t in range(len(Q[i])):
                    Q_dynamic_flatten.append(Q[i][t])
                    #print(np.shape(Q[i][t]))
                    if t==0:
                        self.head_pos_dynamic_flatten[i] = count1
                    self.dynamic_idx_dynamic_flatten.append(count1) 

                    Q_flatten.append(Q[i][t])
                    if t==0:
                        self.head_pos_flatten[i] = count2
                    self.dynamic_to_flatten[count1] = count2
                    count2 += 1 

                    count1 += 1 


        self.Q_dynamic_flatten = Q_dynamic_flatten 
        self.Q_flatten = Q_flatten
        self.num_measure_dynamic_flatten = len(self.static_idx_dynamic_flatten)+len(self.dynamic_idx_dynamic_flatten)
        self.num_measure_flatten = len(self.static_idx_flatten) + len(self.dynamic_idx_flatten)

    def _flatten(self, Q):
        """Update cov matrix and flattened matrix index. 
        Not used any more. Merged to _dynamic_flatten. 
        """

        ### flatten to be cov matrix 
        Q_flatten = []
        self.head_pos_flatten = {}
        self.static_idx_flatten = []
        self.dynamic_idx_flatten = []

        count = 0
        for i in self.num_measure:
            for t in range(len(Q[i])):
                Q_flatten.append(Q[i][t])
                if t==0:
                    self.head_pos_flatten[i] = count 
                if i in self.static_idx:
                    self.static_idx_flatten.append(count)
                else:
                    self.dynamic_idx_flatten.append(count)
                count += 1 

        self.Q_flatten = Q_flatten 

    def _build_sigma(self, error_cov, error_option):
        """Build error covariance matrix 

        if error_cov is None, return an identity matrix 
        option 1: a list, each element is the corresponding variance, a.k.a. diagonal elements.
            Shape: Sum(Nt) 
        option 2: a list of lists, each element is the error covariances
            This option assumes covariances not between measurements, but between timepoints for one measurement
            Shape: Nm * (Nt_m * Nt_m)
        option 3: a list of list, covariance matrix for a single time steps 
            This option assumes the covariances between measurements at the same timestep in a time-invariant way 
            Shape: Nm * Nm
        option 4: a list of list, covariance matrix for the flattened measurements 
            Shape: sum(Nt) * sum(Nt) 
        """
        
        Sigma = np.zeros((self.total_num_time, self.total_num_time))
        # identity matrix 
        if (not error_cov) or (error_option==1):
            if not error_cov:
                error_cov = [1]*self.total_num_time
            for i in range(self.total_num_time):
                Sigma[i,i] = error_cov[i]

        elif error_option == 2: 
            for i in range(self.num_measure):
                # give the error covariance to Sigma 
                sigma_i_start = self.head_pos_flatten[i]
                for t1 in range(self.Nt[i]):
                    for t2 in range(self.Nt[i]):
                        Sigma[sigma_i_start+t1, sigma_i_start+t2] = error_cov[i][t1][t2]

        elif error_option == 3:
            for i in range(self.num_measure):
                for j in range(self.num_measure):
                    cov_ij = error_cov[i,j]
                    head_i = self.head_pos_flatten[i]
                    head_j = self.head_pos_flatten[j]
                    # i, j may have different timesteps
                    for t in range(min(self.Nt[i], self.Nt[j])):
                        Sigma[t+head_i, t+head_j] = cov_ij
     
        elif error_option == 4:
            Sigma = np.asarray(error_cov)

        return Sigma
        

    def _split_sigma(self, Sigma):
        """Split the error covariance matrix to be used for computation
        """
        Sigma_inv = np.linalg.pinv(Sigma)

        self.Sigma_inv = {}
        
        # between static and static: (Nt_i+Nt_j)*(Nt_i+Nt_j) matrix
        for i in self.static_idx_dynamic_flatten:
            for j in self.static_idx_dynamic_flatten:
                sig = np.zeros((self.Nt[i], self.Nt[j]))
                # row [i, i+Nt_i], column [i, i+Nt_i]
                for ti in range(self.Nt[i]):
                    for tj in range(self.Nt[j]):
                        sig[ti, tj] = Sigma_inv[self.head_pos_flatten[i]+ti, self.head_pos_flatten[j]+tj]
                self.Sigma_inv[(i,j)] = sig

        # between static and dynamic: Nt*1 matrix
        for i in self.static_idx_dynamic_flatten:
            for j in self.dynamic_idx_dynamic_flatten:
                sig = np.zeros((self.Nt[i], 1))
                # row [i, i+Nt_i], col [j]
                for t in range(self.Nt[i]):
                    sig[t, 0] = Sigma_inv[self.head_pos_flatten[i]+t, self.dynamic_to_flatten[j]] 
                self.Sigma_inv[(i,j)] = sig

        # between static and dynamic: Nt*1 matrix
        for i in self.dynamic_idx_dynamic_flatten:
            for j in self.static_idx_dynamic_flatten:
                sig = np.zeros((self.Nt[j], 1))
                # row [j, j+Nt_j], col [i]
                for t in range(self.Nt[j]):
                    sig[t, 0] = Sigma_inv[self.head_pos_flatten[j]+t, self.dynamic_to_flatten[i]] 
                self.Sigma_inv[(i,j)] = sig

        # between dynamic and dynamic: a scalar number 
        for i in self.dynamic_idx_dynamic_flatten:
            for j in self.dynamic_idx_dynamic_flatten:
                self.Sigma_inv[(i,j)] = Sigma_inv[i,j]

        
    def fim_computation(self):
        """
        compute a list of FIM. 
        """

        self.fim_collection = []

        for i in range(self.num_measure_dynamic_flatten):
            for j in range(self.num_measure_dynamic_flatten):
                if i in self.static_idx_dynamic_flatten and j in self.static_idx_dynamic_flatten:
                    unit = np.asarray(self.Q_dynamic_flatten[i]).T@self.Sigma_inv[(i,j)]@np.asarray(self.Q_dynamic_flatten[j])
                    
                elif i in self.static_idx_dynamic_flatten and j in self.dynamic_idx_dynamic_flatten:
                    unit = np.asarray(self.Q_dynamic_flatten[i]).T@self.Sigma_inv[(i,j)]@np.asarray(self.Q_dynamic_flatten[j]).reshape(1,self.num_param)

                elif i in self.dynamic_idx_dynamic_flatten and j in self.static_idx_dynamic_flatten:
                    print(np.asarray(self.Q_dynamic_flatten[i]).reshape(1, self.num_param))
                    print(i,j)
                    print(np.asarray(self.Q_dynamic_flatten[j]))
                    unit = np.asarray(self.Q_dynamic_flatten[i]).reshape(1, self.num_param).T@self.Sigma_inv[(i,j)].T@np.asarray(self.Q_dynamic_flatten[j])

                else:
                    unit = self.Sigma_inv[(i,j)]*np.asarray(self.Q_dynamic_flatten[i]).reshape(1, self.num_param).T@np.asarray(self.Q_dynamic_flatten[j]).reshape(1,self.num_param)

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
        m.NumRes = pyo.Set(initialize=range(self.num_measure))
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
                            summi += m.cov_y[i,j]*self.fim_collection[i*self.num_measure+j][a][b]
                        else:
                            summi += m.cov_y[j,i]*self.fim_collection[i*self.num_measure+j][a][b]
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
            return m.TotalDynamic==sum(m.cov_y[i,i] for i in range(num_fixed, self.num_measure))
        
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

        

                




    

