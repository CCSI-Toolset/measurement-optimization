"""
Measurement optimization tool 
@University of Notre Dame
"""
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from greybox_generalize import LogDetModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
from enum import Enum
from idaes.core.util.model_diagnostics import DegeneracyHunter

class covariance_lib(Enum): 
    """Covariance definition 
    if identity: error covariance matrix is an identity matrix
    if variance: a list, each element is the corresponding variance, a.k.a. diagonal elements.
        Shape: Sum(Nt) 
    if time_correlation: a list of lists, each element is the error covariances
        This option assumes covariances not between measurements, but between timepoints for one measurement
        Shape: Nm * (Nt_m * Nt_m)
    if measure_correlation: a list of list, covariance matrix for a single time steps 
        This option assumes the covariances between measurements at the same timestep in a time-invariant way 
        Shape: Nm * Nm
    if time_measure_correlation: a list of list, covariance matrix for the flattened measurements 
        Shape: sum(Nt) * sum(Nt) 
    """
    identity = 0
    variance = 1
    time_correlation = 2 
    measure_correlation = 3
    time_measure_correlation = 4 

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

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
            # jacobian remove fisrt column which is column names
            jaco.append(list(jaco_list[i][1:]))

        self.jaco = jaco

    def get_Q_list(self, static_idx, dynamic_idx, Nt):
        """Combine Q for each measurement to be in one Q.
        measure_info: a pandas dataframe containing measurement information
        Nt: number of timepoints is needed to split Q for each measurement 

        Return 
        Q: jacobian information for main class use
        """
        # number of timepoints for each measurement 
        self.Nt = Nt

        Q = [] 
        # if there is static-cost measurements
        if static_idx is not None:
            for i in static_idx:
                Q.append(self._split_jaco(i))
        # if there is dynamic-cost measurements
        if dynamic_idx is not None:
            for j in dynamic_idx:
                Q.append(self._split_jaco(j))

        return Q 


    def _split_jaco(self, idx):
        """Split Jacobian according to measurements
        idx: idx of measurements

        Return: 
        jaco_idx: a list of jacobian information for one measurement 
        """
        jaco_idx = self.jaco[idx*self.Nt:(idx+1)*self.Nt][:]
        return jaco_idx 
    


class MeasurementOptimizer:
    def __init__(self, Q, measure_info, error_cov=None, error_opt=covariance_lib.identity, verbose=True):
        """
        Argument
        --------
        :param Q: a list of lists containing Jacobian matrix. 
            It contains m lists, m is the No. of meausrements 
            Each list contains an N_t_m*num_param elements, which is the sensitivity matrix Q for measurement m 
        :param measure_info: a pandas DataFrame containing measurement information.
        :param error_cov: 
            defined error covariance matrix here
            if covariance_lib.identity: error covariance matrix is an identity matrix
            if covariance_lib.variance: a list, each element is the corresponding variance, a.k.a. diagonal elements.
                Shape: Sum(Nt) 
            if covariance_lib.time_correlation: a list of lists, each element is the error covariances
                This option assumes covariances not between measurements, but between timepoints for one measurement
                Shape: Nm * (Nt_m * Nt_m)
            if covariance_lib.measure_correlation: a list of list, covariance matrix for a single time steps 
                This option assumes the covariances between measurements at the same timestep in a time-invariant way 
                Shape: Nm * Nm
            if covariance_lib.time_measure_correlation: a list of list, covariance matrix for the flattened measurements 
                Shape: sum(Nt) * sum(Nt) 
        :param: error_opt:
            can choose from None, 1, 2, 3, 4. See above comments.
        :param verbose: if print debug sentences
        """
        # # of static and dynamic measurements
        static_idx = measure_info[measure_info['dynamic_cost']==0].index.values.tolist()
        dynamic_idx = measure_info[measure_info['dynamic_cost']!=0].index.values.tolist()
        # store static and dynamic measurements dataframe 
        self.dynamic_cost_measure_info = measure_info[measure_info['dynamic_cost']!=0]
        self.static_cost_measure_info = measure_info[measure_info['dynamic_cost']==0]

        self.measure_info = measure_info
        self.num_static = len(static_idx)
        self.static_idx = static_idx
        self.num_dynamic = len(dynamic_idx)
        self.dynamic_idx = dynamic_idx
        self.num_measure = len(Q)
        assert self.num_measure==self.num_dynamic+self.num_static

        # measurements can have different # of timepoints
        # Nt key: measurement index, value: # of timepoints for this measure
        self.Nt = {}
        for i in range(self.num_measure):
            self.Nt[i] = len(Q[i])
        # total number of all measurements and all time points
        self.total_num_time = sum(self.Nt.values())

        self.num_param = len(Q[0][0])
        self.verbose = verbose
        self.measure_name = measure_info['name'].tolist() # measurement name list 
        self.cost_list = self.static_cost_measure_info['static_cost'].tolist() # static measurements list
        # add dynamic-cost measurements list 
        for i in dynamic_idx:
            q_ind = measure_info.iloc[i]['Q_index']
            # loop over dynamic-cost measurements time points
            for t in range(self.Nt[q_ind]):
                self.cost_list.append(measure_info.iloc[i]['dynamic_cost'])

        # dynamic-cost measurements install cost
        self.dynamic_install_cost = self.dynamic_cost_measure_info['static_cost'].tolist()

        # min time interval, only for dynamic-cost measurements
        min_time_interval = measure_info['min_time_interval'].tolist()
        # if a minimal time interval is set up 
        if np.asarray(min_time_interval).any():
            self.min_time_interval = min_time_interval
        else:
            # this option can also be None, means there are no time interval limitation
            self.min_time_interval = None

        # each manual number 
        each_manual_number = measure_info['max_manual_number'].tolist()
        if np.asarray(each_manual_number).any():
            self.each_manual_number = each_manual_number
        else:
            self.each_manual_number = None 

        # flattened Q and indexes
        self._dynamic_flatten(Q)

        # build and check PSD of Sigma
        Sigma = self._build_sigma(error_cov, error_opt)
        self._split_sigma(Sigma)

    def _dynamic_flatten(self, Q):
        """Update dynamic flattened matrix index. 
        dynamic_flatten matrix: flatten dynamic-cost measurements, not flatten static-costs, [s1, d1|t1, ..., d1|tN, s2]
        Flatten matrix: flatten dynamic-cost and static-cost measuremenets
        """

        ### dynamic_flatten: to be decision matrix 
        Q_dynamic_flatten = []
        # position index in Q_dynamic_flatten where each measurement starts
        self.head_pos_dynamic_flatten = {}
        # all static measurements index after dynamic_flattening
        self.static_idx_dynamic_flatten = []
        self.dynamic_idx_dynamic_flatten = []

        ### flatten: flatten all measurement all costs 
        Q_flatten = []
        # position index in Q_flatten where each measurement starts
        self.head_pos_flatten = {}
        # all static measurements index after flatten
        self.static_idx_flatten = []
        # all dynamic measurements index after flatten
        self.dynamic_idx_flatten = []

        # map dynamic index to flatten index 
        # key: dynamic index, value: corresponding indexes in flatten matrix. For static, it's a list. For dynamic, it's a index value
        self.dynamic_to_flatten = {}

        # counter for dynamic_flatten
        count1 = 0
        # counter for flatten
        count2 = 0
        for i in range(self.num_measure):
            if i in self.static_idx: # static measurements are not flattened for dynamic flatten
                # dynamic_flatten
                Q_dynamic_flatten.append(Q[i])
                self.head_pos_dynamic_flatten[i] = count1 
                self.static_idx_dynamic_flatten.append(count1)
                self.dynamic_to_flatten[count1] = [] # static measurement's dynamic_flatten index corresponds to a list of flattened index

                # flatten 
                for t in range(len(Q[i])):
                    Q_flatten.append(Q[i][t])
                    if t==0:
                        self.head_pos_flatten[i] = count2
                    self.static_idx_flatten.append(count2)
                    # map all timepoints to the dynamic_flatten static index
                    self.dynamic_to_flatten[count1].append(count2)
                    count2 += 1 

                count1 += 1 

            else:
                # dynamic measurements are flattend for both dynamic_flatten and flatten
                for t in range(len(Q[i])):
                    Q_dynamic_flatten.append(Q[i][t])
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
        # dimension after dynamic_flatten
        self.num_measure_dynamic_flatten = len(self.static_idx_dynamic_flatten)+len(self.dynamic_idx_dynamic_flatten)
        # dimension after flatten
        self.num_measure_flatten = len(self.static_idx_flatten) + len(self.dynamic_idx_flatten)

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
        if (error_option==covariance_lib.identity) or (error_option==covariance_lib.variance):
            if not error_cov:
                error_cov = [1]*self.total_num_time
            # loop over diagonal elements and change
            for i in range(self.total_num_time):
                Sigma[i,i] = error_cov[i]

        elif error_option == covariance_lib.time_correlation: 
            for i in range(self.num_measure):
                # give the error covariance to Sigma 
                sigma_i_start = self.head_pos_flatten[i]
                # loop over all timepoints for measurement i 
                for t1 in range(self.Nt[i]):
                    for t2 in range(self.Nt[i]):
                        Sigma[sigma_i_start+t1, sigma_i_start+t2] = error_cov[i][t1][t2]

        elif error_option == covariance_lib.measure_correlation:
            print("Covariances are between measurements at the same time.")
            for i in range(self.num_measure):
                for j in range(self.num_measure):
                    cov_ij = error_cov[i][j]
                    head_i = self.head_pos_flatten[i]
                    head_j = self.head_pos_flatten[j]
                    # i, j may have different timesteps
                    for t in range(min(self.Nt[i], self.Nt[j])):
                        Sigma[t+head_i, t+head_j] = cov_ij 
     
        elif error_option == covariance_lib.time_measure_correlation:
            Sigma = np.asarray(error_cov)

        self.Sigma = Sigma

        return Sigma
        

    def _split_sigma(self, Sigma):
        """Split the error covariance matrix to be used for computation
        """
        # Inverse of covariance matrix is used 
        Sigma_inv = np.linalg.pinv(Sigma)
        self.Sigma_inv_matrix = Sigma_inv
        # Use a dicionary to store the inverse of sigma as either scalar number, vector, or matrix
        self.Sigma_inv = {}
        
        # between static and static: (Nt_i+Nt_j)*(Nt_i+Nt_j) matrix
        for i in self.static_idx_dynamic_flatten: # loop over static measurement index
            for j in self.static_idx_dynamic_flatten: # loop over static measurement index 
                # should be a (Nt_i+Nt_j)*(Nt_i+Nt_j) matrix
                sig = np.zeros((self.Nt[i], self.Nt[j]))
                # row [i, i+Nt_i], column [i, i+Nt_i]
                for ti in range(self.Nt[i]): # loop over time points 
                    for tj in range(self.Nt[j]): # loop over time points
                        sig[ti, tj] = Sigma_inv[self.head_pos_flatten[i]+ti, self.head_pos_flatten[j]+tj]
                self.Sigma_inv[(i,j)] = sig

        # between static and dynamic: Nt*1 matrix
        for i in self.static_idx_dynamic_flatten: # loop over static measurement index
            for j in self.dynamic_idx_dynamic_flatten: # loop over dynamic measuremente index 
                # should be a vector, here as a Nt*1 matrix
                sig = np.zeros((self.Nt[i], 1))
                # row [i, i+Nt_i], col [j]
                for t in range(self.Nt[i]): # loop over time points 
                    sig[t, 0] = Sigma_inv[self.head_pos_flatten[i]+t, self.dynamic_to_flatten[j]] 
                self.Sigma_inv[(i,j)] = sig

        # between static and dynamic: Nt*1 matrix
        for i in self.dynamic_idx_dynamic_flatten: # loop over dynamic measurement index 
            for j in self.static_idx_dynamic_flatten: # loop over static measurement index
                # should be a vector, here as Nt*1 matrix 
                sig = np.zeros((self.Nt[j], 1)) 
                # row [j, j+Nt_j], col [i]
                for t in range(self.Nt[j]): # loop over time 
                    sig[t, 0] = Sigma_inv[self.head_pos_flatten[j]+t, self.dynamic_to_flatten[i]] 
                self.Sigma_inv[(i,j)] = sig

        # between dynamic and dynamic: a scalar number 
        for i in self.dynamic_idx_dynamic_flatten: # loop over dynamic measurement index
            for j in self.dynamic_idx_dynamic_flatten: # loop over dynamic measurement index 
                # should be a scalar number 
                self.Sigma_inv[(i,j)] = Sigma_inv[self.dynamic_to_flatten[i],self.dynamic_to_flatten[j]]

        
    def fim_computation(self):
        """
        compute a list of FIM. 
        """

        self.fim_collection = []

        for i in range(self.num_measure_dynamic_flatten):
            for j in range(self.num_measure_dynamic_flatten):

                # static*static
                if i in self.static_idx_dynamic_flatten and j in self.static_idx_dynamic_flatten:
                    #print("static * static, cov:", self.Sigma_inv[(i,j)])
                    unit = np.asarray(self.Q_dynamic_flatten[i]).T@self.Sigma_inv[(i,j)]@np.asarray(self.Q_dynamic_flatten[j])
                    
                # static*dynamic
                elif i in self.static_idx_dynamic_flatten and j in self.dynamic_idx_dynamic_flatten:
                    #print("static*dynamic, cov:", self.Sigma_inv[(i,j)])
                    unit = np.asarray(self.Q_dynamic_flatten[i]).T@self.Sigma_inv[(i,j)]@np.asarray(self.Q_dynamic_flatten[j]).reshape(1,self.num_param)

                # static*dynamic
                elif i in self.dynamic_idx_dynamic_flatten and j in self.static_idx_dynamic_flatten:
                    #print("static*dynamic, cov:", self.Sigma_inv[(i,j)])
                    unit = np.asarray(self.Q_dynamic_flatten[i]).reshape(1, self.num_param).T@self.Sigma_inv[(i,j)].T@np.asarray(self.Q_dynamic_flatten[j])

                # dynamic*dynamic
                else:
                    #print("dynamic*dynamic, cov:", self.Sigma_inv[(i,j)])
                    unit = self.Sigma_inv[(i,j)]*np.asarray(self.Q_dynamic_flatten[i]).reshape(1, self.num_param).T@np.asarray(self.Q_dynamic_flatten[j]).reshape(1,self.num_param)

                self.fim_collection.append(unit.tolist())

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
        Analyze one given FIM
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

    def continuous_optimization(self, mixed_integer=False, obj="A", 
                                fix=False, sparse=False,
                                num_dynamic_t_name = None, 
                                manual_number=20, budget=100, 
                                init_cov_y=None, initial_fim=None,
                                dynamic_install_initial = None,
                                static_dynamic_pair=None,
                                time_interval_all_dynamic=False, 
                                total_manual_num_init=10):
        
        """Continuous optimization problem formulation. 

        Parameter 
        ---------
        :param cost_list: A list, containing the cost of each timepoint of corresponding measurement
        :param mixed_integer: not relaxing integer decisions
        :param obj: "A" or "D" optimality, use trace or determinant of FIM 
        :param fix: if solving as a square problem or with DOFs 
        :param sparse: if using sparse set to define decisions and FIM, or not 
        :param num_dynamic_t_name: a list of the exact time points for the dynamic-cost measurements time points 
        :param discretize_time: a scalar number of the time interval between two manual measurements 
        :param end_time: a scalar number, the end time of measurements 
        :param manual_number: the maximum number of human measurements for dynamic measurements
        :param budget: total budget
        :param dynamic_install_cost: a list of the installation costs for dynamic-cost measurements 
        :param each_manual_number: the maximum number of human measurements for each dynamic-cost measurement 
        :param init_cov_y: initialize decision variables 
        :param initial_fim: initialize FIM
        :param dynamic_install_initial: initialize if_dynamic_install
        :param static_dynamic_pair: a list of the name of measurements, that are selected as either dynamic or static measurements. 
        """

        m = pyo.ConcreteModel()

        # response set
        m.NumRes = pyo.Set(initialize=range(self.num_measure_dynamic_flatten))
        # FIM set 
        m.DimFIM = pyo.Set(initialize=range(self.num_param))

        # dynamic measurements parameters 
        # dynamic measurement number of timepoints 
        self.dynamic_Nt = self.Nt[self.num_static]
        # dynamic measurement index number 
        m.DimDynamic = pyo.Set(initialize=range(self.num_static, self.num_measure))
        # turn dynamic measurement number of timepoints into a pyomo set 
        m.DimDynamic_t = pyo.Set(initialize=range(self.dynamic_Nt)) 
        
        dynamic_time = {}
        for i in range(self.dynamic_Nt):
            dynamic_time[i] = num_dynamic_t_name[i]
        self.dynamic_time = dynamic_time

        # initialize with identity
        def identity(m,a,b):
            return 1 if a==b else 0 
        def initialize_point(m,a,b):
            return init_cov_y[a][b]
        
        if init_cov_y.any():
            initialize=initialize_point
        else:
            initialize=identity

        # sparse NumRes, only define the upper triangle of symmetric matrices 
        def NumReshalf_init(m):
            return ((a,b) for a in m.NumRes for b in range(a, self.num_measure_dynamic_flatten))
        
        def DimFIMhalf_init(m):
            return ((a,b) for a in m.DimFIM for b in range(a, self.num_param))
        
        m.NumRes_half = pyo.Set(dimen=2, initialize=NumReshalf_init)
        m.DimFIM_half = pyo.Set(dimen=2, initialize=DimFIMhalf_init)
        
        # decision variables
        if mixed_integer:
            if sparse:
                m.cov_y = pyo.Var(m.NumRes_half, initialize=initialize, bounds=(0,1), within=pyo.Binary)
            else:
                m.cov_y = pyo.Var(m.NumRes, m.NumRes, initialize=initialize, bounds=(0,1), within=pyo.Binary)
        else:
            if sparse:
                m.cov_y = pyo.Var(m.NumRes_half, initialize=initialize, bounds=(0,1), within=pyo.NonNegativeReals)
            else:
                m.cov_y = pyo.Var(m.NumRes, m.NumRes, initialize=initialize, bounds=(0,1), within=pyo.NonNegativeReals)
        
        # use a fix option to compute results for square problems with given y 
        if fix:
            m.cov_y.fix()

        def init_fim(m,p,q):
            return initial_fim[p,q]
        
        if initial_fim is not None:
            # Initialize dictionary for grey-box model
            fim_initial_dict = {}
            for i in range(self.num_param):
                for j in range(i, self.num_param):
                    str_name = 'ele_'+str(i)+"_"+str(j)
                    fim_initial_dict[str_name] = initial_fim[i,j] 
        
        if sparse:
            m.TotalFIM = pyo.Var(m.DimFIM_half, initialize=init_fim)
        else:
            m.TotalFIM = pyo.Var(m.DimFIM, m.DimFIM, initialize=init_fim)

        ### compute FIM 
        def eval_fim(m, a, b):
            """
            Evaluate fim 

            a, b: dimensions for FIM, iterate in parameter set 
            """
            if a <= b: 
                summi = 0 
                for i in m.NumRes:
                    for j in m.NumRes:
                        if j>=i:
                            summi += m.cov_y[i,j]*self.fim_collection[i*self.num_measure_dynamic_flatten+j][a][b]
                        else:
                            summi += m.cov_y[j,i]*self.fim_collection[i*self.num_measure_dynamic_flatten+j][a][b]
                return m.TotalFIM[a,b] == summi
            else:
                return m.TotalFIM[a,b] == m.TotalFIM[b,a]
            
    
        def total_dynamic(m):
            return m.TotalDynamic==sum(m.cov_y[i,i] for i in range(self.num_static, self.num_measure_dynamic_flatten))
            
        ### cov_y constraints
        def y_covy1(m, a, b):
            if b > a:
                return m.cov_y[a, b] <= m.cov_y[a, a]
            else:
                # skip lower triangle constraints since we don't define them 
                return pyo.Constraint.Skip
            
        def y_covy2(m, a, b):
            if b > a:
                return m.cov_y[a, b] <= m.cov_y[b, b]
            else:
                # skip lower triangle constraints since we don't define them 
                return pyo.Constraint.Skip
            
        def y_covy3(m, a, b):
            if b > a:
                return m.cov_y[a, b] >= m.cov_y[a, a] + m.cov_y[b, b] - 1
            else:
                # skip lower triangle constraints since we don't define them 
                return pyo.Constraint.Skip
            
        def symmetry(m,a,b):
            if a<b:
                return m.cov_y[a,b] == m.cov_y[b,a]
            else:
                # skip lower triangle constraints since we don't define them 
                return pyo.Constraint.Skip

        ### cost constraints
        def cost_compute(m):
            """Compute cost
            cost = static-cost measurement cost + dynamic-cost measurement installation cost + dynamic-cost meausrement timepoint cost 
            """
            return m.cost == sum(m.cov_y[i,i]*self.cost_list[i] for i in m.NumRes)+sum(m.if_install_dynamic[j]*self.dynamic_install_cost[j-self.num_static] for j in m.DimDynamic)
        
        def cost_limit(m):
            return m.cost <= budget 

        def total_dynamic_con(m):
            return m.TotalDynamic<=manual_number
        
        def dynamic_fix_yd(m,i,j):
            # map measurement index i to its dynamic_flatten index
            start = self.num_static + (i-self.num_static)*self.dynamic_Nt+j
            return m.if_install_dynamic[i] >= m.cov_y[start,start]
        
        def dynamic_fix_yd_con2(m,i):
            start = self.num_static + (i-self.num_static)*self.dynamic_Nt
            end = self.num_static + (i-self.num_static+1)*self.dynamic_Nt
            return m.if_install_dynamic[i] <= sum(m.cov_y[j,j] for j in range(start, end))
        
        # set up Design criterion
        def ComputeTrace(m):
            sum_x = sum(m.TotalFIM[j,j] for j in m.DimFIM)
            return sum_x

        # add constraints
        if sparse:
            m.TotalFIM_con = pyo.Constraint(m.DimFIM_half, rule=eval_fim)
        else:
            m.TotalFIM_con = pyo.Constraint(m.DimFIM, m.DimFIM, rule=eval_fim)
        
        if not fix: 

            m.TotalDynamic = pyo.Var(initialize=total_manual_num_init)
            m.manual = pyo.Constraint(rule=total_dynamic)
        
            if mixed_integer and not sparse:
                m.sym = pyo.Constraint(m.NumRes, m.NumRes, rule=symmetry)
            
            m.cov1 = pyo.Constraint(m.NumRes, m.NumRes, rule=y_covy1)
            m.cov2 = pyo.Constraint(m.NumRes, m.NumRes, rule=y_covy2)
            m.cov3 = pyo.Constraint(m.NumRes, m.NumRes, rule=y_covy3)
                
            m.con_manual = pyo.Constraint(rule=total_dynamic_con)

            # dynamic-cost measurements installaction cost 
            def dynamic_install_init(m,j):
                if dynamic_install_initial is None:
                    return 0
                else:
                    print(j)
                    print(dynamic_install_initial)
                    return dynamic_install_initial[j-self.num_static]

            if mixed_integer:
                m.if_install_dynamic = pyo.Var(m.DimDynamic, initialize=dynamic_install_init, bounds=(0,1), within=pyo.Binary)
            else:
                m.if_install_dynamic = pyo.Var(m.DimDynamic, initialize=dynamic_install_init, bounds=(0,1))
            m.dynamic_cost = pyo.Constraint(m.DimDynamic, m.DimDynamic_t, rule=dynamic_fix_yd)
            m.dynamic_con2 = pyo.Constraint(m.DimDynamic, rule=dynamic_fix_yd_con2)

            # each manual number smaller than 5 
            if self.each_manual_number is not None:
                
                for i in range(self.num_dynamic):
                    def dynamic_manual_num(m):
                        start = self.num_static + i*self.dynamic_Nt
                        end = self.num_static + (i+1)*self.dynamic_Nt
                        cost = sum(m.cov_y[j,j] for j in range(start, end))
                        return cost <= self.each_manual_number[0]
                    
                    con_name = "con"+str(i)
                    m.add_component(con_name, pyo.Constraint(expr=dynamic_manual_num))

            if static_dynamic_pair is not None: 
                for i, pair in enumerate(static_dynamic_pair):
                    def static_dynamic_pair_con(m):
                        return m.if_install_dynamic[pair[1]]+m.cov_y[pair[0],pair[0]] <= 1
                    
                    con_name = "con_sta_dyn"+str(i)
                    m.add_component(con_name, pyo.Constraint(expr=static_dynamic_pair_con))

            if self.min_time_interval is not None:
                if time_interval_all_dynamic: 
                    for t in range(self.dynamic_Nt):
                        # end time is an open end of the region, so another constraint needs to be added to include end_time
                        #if dynamic_time[t]+discretize_time <= end_time+0.1*discretize_time:                 
                        def discretizer(m):
                            sumi = 0
                            count = 0 
                            while (count+t<self.dynamic_Nt) and (dynamic_time[count+t]-dynamic_time[t])<self.min_time_interval[0]:
                                for i in m.DimDynamic:
                                    surro_idx = self.num_static + (i-self.num_static)*self.dynamic_Nt + t + count
                                    sumi += m.cov_y[surro_idx, surro_idx]
                                count += 1 

                            return sumi <= 1 

                        con_name="con_discreti_"+str(i)+str(t)
                        m.add_component(con_name, pyo.Constraint(expr=discretizer))
                else:
                    for i in m.DimDynamic:
                        for t in range(self.dynamic_Nt):
                            # end time is an open end of the region, so another constraint needs to be added to include end_time
                            #if dynamic_time[t]+discretize_time <= end_time+0.1*discretize_time:       
                                                    
                            def discretizer(m):
                                sumi = 0

                                count = 0 
                                while (count+t<self.dynamic_Nt) and (dynamic_time[count+t]-dynamic_time[t])<self.min_time_interval[0]:
                                    surro_idx = self.num_static + (i-self.num_static)*self.dynamic_Nt + t + count
                                    sumi += m.cov_y[surro_idx, surro_idx]
                                    count += 1 

                                return sumi <= 1 

                            con_name="con_discreti_"+str(i)+str(t)
                            m.add_component(con_name, pyo.Constraint(expr=discretizer))
                        
            
            m.cost = pyo.Var(initialize=budget)
            m.cost_compute = pyo.Constraint(rule=cost_compute)
            m.budget_limit = pyo.Constraint(rule=cost_limit)

        # set objective 
        if obj == "A":
            m.Obj = pyo.Objective(rule=ComputeTrace, sense=pyo.maximize)

        elif obj == "D":

            def _model_i(b):
                self.build_model_external(b, fim_init=fim_initial_dict)
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


    def build_model_external(self, m, fim_init=None):
        ex_model = LogDetModel(num_para=self.num_param, init_fim=fim_init)
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)

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
    
    def solve(self, mod, mip_option=False, objective="A", degeneracy_hunter=False):
        if not mip_option and objective=="A":
            solver = pyo.SolverFactory('ipopt')
            solver.options['linear_solver'] = "ma57"
            if degeneracy_hunter:
                solver.options['max_iter'] = 0
                solver.options['bound_push'] = 1E-6
            solver.solve(mod, tee=True)
            if degeneracy_hunter:
                dh = DegeneracyHunter(mod, solver=solver)

        elif mip_option and objective=="A":
            solver = pyo.SolverFactory('gurobi', solver_io="python")
            #solver.options['mipgap'] = 0.1
            solver.solve(mod, tee=True)
            
        elif objective=="D":  
            solver = pyo.SolverFactory('cyipopt')
            solver.config.options['hessian_approximation'] = 'limited-memory' 
            additional_options={'max_iter':3000, 'output_file': 'console_output',
                                'linear_solver':'mumps'}
            
            for k,v in additional_options.items():
                solver.config.options[k] = v
            solver.solve(mod, tee=True)
            
        if degeneracy_hunter:
            return mod, dh
        else:
            return mod

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
            

    def extract_solutions(self, mod):
        """
        Extract and show solutions. 
        """
        # ans_y is a list of lists
        ans_y = np.zeros((self.num_measure_dynamic_flatten,self.num_measure_dynamic_flatten))

        for i in range(self.num_measure_dynamic_flatten):
            for j in range(i, self.num_measure_dynamic_flatten):
                cov = pyo.value(mod.cov_y[i,j])
                ans_y[i,j] = ans_y[j,i] = cov 

        # round small errors
        for i in range(len(ans_y)):
            for j in range(len(ans_y[0])):
                if ans_y[i][j] < 0.05:
                    ans_y[i][j] = int(0)
                elif ans_y[i][j] > 0.95:
                    ans_y[i][j] = int(1)
                else: 
                    ans_y[i][j] = round(ans_y[i][j], 2)

        for c in range(self.num_static):
            print(self.measure_name[c], ": ", ans_y[c,c])

        sol_y = np.asarray([ans_y[i,i] for i in range(self.num_static, self.num_measure_dynamic_flatten)])

        sol_y = np.reshape(sol_y, (self.num_dynamic, self.dynamic_Nt))
        np.around(sol_y)

        for r in range(len(sol_y)):
            #print(dynamic_name[r], ": ", sol_y[r])
            print(self.measure_name[r+self.num_static])
            print(sol_y[r])
            #for i, t in enumerate(sol_y[r]):
            #    if t>0.5:
            #        print(self.dynamic_time[i])

        return ans_y, sol_y



                




    

