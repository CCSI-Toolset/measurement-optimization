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
#from idaes.core.util.model_diagnostics import DegeneracyHunter

class CovarianceStructure(Enum): 
    """Covariance definition 
    if identity: error covariance matrix is an identity matrix
    if variance: a numpy vector, each element is the corresponding variance, a.k.a. diagonal elements.
        Shape: Sum(Nt) 
    if time_correlation: a 3D numpy array, each element is the error covariances
        This option assumes covariances not between measurements, but between timepoints for one measurement
        Shape: Nm * (Nt_m * Nt_m)
    if measure_correlation: a 2D numpy array, covariance matrix for a single time steps 
        This option assumes the covariances between measurements at the same timestep in a time-invariant way 
        Shape: Nm * Nm
    if time_measure_correlation: a 2D numpy array, covariance matrix for the flattened measurements 
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
    
class ObjectiveLib(Enum):
    """
    Objective function library
    
    if A: minimize the trace of FIM
    if D: minimize the determinant of FIM
    """
    A = 0 
    D = 1 
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class DataProcess:
    """Data processing class. Only process a certain format of CSV file."""
    def __init__(self) -> None:
        return 
    
    def read_jacobian(self, filename):
        """Read jacobian from csv file 

        Arguments
        ---------
        :param filename: a string of the csv file name

        This csv file should have the following format:
        columns: parameters to be estimated 
        (An extra first column is added for index)
        rows: measurement timepoints
        data: jacobian values

        The csv file example: 

        Column index  |  Parameter 1 | Parameter 2 |  ...  | Parameter P 
        measurement 1 |    number    |  number     |  ...  | number   
        measurement 2 |    number    |  number     |  ...  | number  
        ...
        measurement N |    number    |  number     |  ...  | number   

        Number according to measurement i, parameter j is 
        the gradient of measurement i w.r.t parameter j 

        Returns
        ------
        None
        """
        jacobian_info = pd.read_csv(filename, index_col=False)
        # it needs to be converted to numpy array or it gives error
        jacobian_list = np.asarray(jacobian_info) 

        # jacobian_list (N*Np matrix) is converted to a list of lists (each list is a [Np*1] vector)
        # to separate different measurements
        # because FIM requires different gradient [Np*1] vectors multiply together
        jacobian = []
        # first columns are parameter names in string, so to be removed
        for i in range(len(jacobian_list)):
            # jacobian remove first column which is column index. see doc string for example input
            jacobian.append(list(jacobian_list[i][1:]))

        self.jacobian = jacobian

    def get_Q_list(self, static_measurement_idx, dynamic_measurement_idx, Nt):
        """Combine Q for each measurement to be in one Q.
        Q is a list of lists containing jacobian matrix.
        each list contains an Nt*n_parameters elements, which is the sensitivity matrix Q for measurement m

        Arguments
        ---------
        :param static_measurement_idx: list of the index for static measurements 
        :param dynamic_measurement_idx: list of the index for dynamic measurements
        :param Nt: number of timepoints is needed to split Q for each measurement 

        Returns
        ------- 
        Q: jacobian information for main class use, a 2D numpy array
        """
        # number of timepoints for each measurement 
        self.Nt = Nt

        # get the maximum index from index set 
        # why not sum up static index number and dynamic index number? because they can overlap 
        max_measure_idx = max(max(static_measurement_idx), max(dynamic_measurement_idx))
        # the total rows of Q should be equal or more than the number of maximum index given by the argument 
        assert(len(Q)>=max_measure_idx*self.Nt), "Inconsistent Jacobian matrix shape. Expecting at least "+str(max_measure_idx*self.Nt)+" rows in Q matrix."

        # initialize Q as a list of lists, after stacking all jacobians, it is converted to numpy array
        # after spliting the overall Jacobian to separate Jacobians for each measurement by self._split_jacobian
        # each separate Jacobians becomes a list of lists
        # here we stack the Q according to the orders the user provides SCM and DCM index
        # Q: jacobian matrix of shape N_measure * Np
        Q = [] 
        # if there is static-cost measurements
        if static_measurement_idx is not None:
            # loop over SCM indices
            for i in static_measurement_idx:
                Q.append(self._split_jacobian(i))
        # if there is dynamic-cost measurements
        if dynamic_measurement_idx is not None:
            # loop over DCM indices
            for j in dynamic_measurement_idx:
                Q.append(self._split_jacobian(j))

        return np.asarray(Q) 


    def _split_jacobian(self, idx):
        """Split jacobian according to measurements
        It splits the overall stacked Q matrix to 
        Q for measurement 1, Q for measurement 2, ..., Q for measurement N 

        Arguments
        ---------
        idx: idx of measurements

        Returns
        -------
        jacobian_idx: a Nt*Np matrix, Nt is the number of timepoints for the measurement. 
            jacobian information for one measurement 
            they are slicing indices for jacobian matrix
        """
        # get a Nt*Np matrix, Nt is the number of timepoint for the measurement 
        jacobian_idx = self.jacobian[idx*self.Nt:(idx+1)*self.Nt][:]
        return jacobian_idx 
    


class MeasurementOptimizer:
    def __init__(self, Q, measure_info, error_cov=None, error_opt=CovarianceStructure.identity, verbose=True):
        """
        Arguments
        ---------
        :param Q: a 2D numpy array
            containing jacobian matrix. 
            It contains m lists, m is the number of meausrements 
            Each list contains an N_t_m*n_parameters elements, which is the sensitivity matrix Q for measurement m 
        :param measure_info: a pandas DataFrame 
            containing measurement information.
            columns: ['name', 'Q_index', 'dynamic_cost', 'static_cost', 'min_time_interval', 'max_manual_number']
        :param error_cov: a numpy array
            defined error covariance matrix here
            if CovarianceStructure.identity: error covariance matrix is an identity matrix
            if CovarianceStructure.variance: a numpy vector, each element is the corresponding variance, a.k.a. diagonal elements.
                Shape: Sum(Nt) 
            if CovarianceStructure.time_correlation: a 3D numpy array, each element is the error covariances
                This option assumes covariances not between measurements, but between timepoints for one measurement
                Shape: Nm * (Nt_m * Nt_m)
            if CovarianceStructure.measure_correlation: a 2D numpy array, covariance matrix for a single time steps 
                This option assumes the covariances between measurements at the same timestep in a time-invariant way 
                Shape: Nm * Nm
            if CovarianceStructure.time_measure_correlation: a 2D numpy array, covariance matrix for the flattened measurements 
                Shape: sum(Nt) * sum(Nt) 
        :param: error_opt: CovarianceStructure
            can choose from identity, variance, time_correlation, measure_correlation, time_measure_correlation. See above comments.
        :param verbose: if print debug sentences

        Returns
        -------
        None 
        """
        # # of static and dynamic measurements
        static_measurement_idx = measure_info[measure_info['dynamic_cost']==0].index.values.tolist()
        dynamic_measurement_idx = measure_info[measure_info['dynamic_cost']!=0].index.values.tolist()
        # store static and dynamic measurements dataframe 
        self.dynamic_cost_measure_info = measure_info[measure_info['dynamic_cost']!=0]
        self.static_cost_measure_info = measure_info[measure_info['dynamic_cost']==0]

        self.measure_info = measure_info
        # check measure_info
        self._check_measure_info()  
        self.n_static_measurements = len(static_measurement_idx)
        self.static_measurement_idx = static_measurement_idx
        self.n_dynamic_measurements = len(dynamic_measurement_idx)
        self.dynamic_measurement_idx = dynamic_measurement_idx
        self.n_total_measurements = len(Q)

        # the Jacobian matrix should have the same rows as the number of DCMs + number of SCMs
        assert self.n_total_measurements==self.n_dynamic_measurements+self.n_static_measurements, \
            "Jacobian matrix does not agree to measurement indices, expecting " + str(len(Q)) + " total measurements in Jacobian." 

        # measurements can have different # of timepoints
        # Nt key: measurement index, value: # of timepoints for this measure
        self.Nt = {}
        for i in range(self.n_total_measurements):
            self.Nt[i] = len(Q[i])
        # total number of all measurements and all time points
        self.total_num_time = sum(self.Nt.values())

        self.n_parameters = len(Q[0][0])
        self.verbose = verbose
        self.measure_name = measure_info['name'].tolist() # measurement name list 
        self.cost_list = self.static_cost_measure_info['static_cost'].tolist() # static measurements list
        # add dynamic-cost measurements list 
        # loop over DCM index list
        for i in dynamic_measurement_idx:
            q_ind = measure_info.iloc[i]['Q_index']
            # loop over dynamic-cost measurements time points
            for _ in range(self.Nt[q_ind]):
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

        # each manual number, for one measurement, how many time points can be chosen at most 
        each_manual_number = measure_info['max_manual_number'].tolist()
        # if there is a value, this is how many time points can be chosen for it. 
        if np.asarray(each_manual_number).any():
            self.each_manual_number = each_manual_number
        # if this is a null list, there is no limitation for how many time points can be chosen for it. 
        else:
            self.each_manual_number = None 

        # flattened Q and indexes
        self._dynamic_flatten(Q)

        # build and check PSD of Sigma
        # check sigma inputs 
        self._check_sigma(error_cov, error_opt)
        
        # build the Sigma and Sigma_inv (error covariance matrix and its inverse matrix)
        Sigma = self._build_sigma(error_cov, error_opt)

        # split Sigma_inv to DCM-DCM error, DCM-SCM error vector, SCM-SCM error matrix 
        self._split_sigma(Sigma)


    def _check_measure_info(self):
        """Check if the measure_info dataframe is successfully built with all information
        """
        if "name" not in self.measure_info:
            raise ValueError("measure_info must have a column named 'name'")
        if "Q_index" not in self.measure_info:
            raise ValueError("measure_info must have a column named 'Q_index'")
        if "dynamic_cost" not in self.measure_info:
            raise ValueError("measure_info must have a column named 'dynamic_cost'")
        if "static_cost" not in self.measure_info:
            raise ValueError("measure_info must have a column named 'static_cost'")
        if "min_time_interval" not in self.measure_info:
            raise ValueError("measure_info must have a column named 'min_time_interval'")
        if "max_manual_number" not in self.measure_info:
            raise ValueError("measure_info must have a column named 'max_manual_number'")
        
    def _check_sigma(self, error_cov, error_option):
        """ Check sigma inputs shape and values

        Arguments
        ---------
        :param error_cov: if error_cov is None, return an identity matrix 
        option 1: a numpy vector, each element is the corresponding variance, a.k.a. diagonal elements.
            Shape: Sum(Nt) 
        option 2: a 3D numpy array, each element is the error covariances
            This option assumes covariances not between measurements, but between timepoints for one measurement
            Shape: Nm * (Nt_m * Nt_m)
        option 3: a 2D numpy array, covariance matrix for a single time steps 
            This option assumes the covariances between measurements at the same timestep in a time-invariant way 
            Shape: Nm * Nm
        option 4: a 2D numpy array, covariance matrix for the flattened measurements 
            Shape: sum(Nt) * sum(Nt) 
        :param: error_opt: CovarianceStructure
            can choose from identity, variance, time_correlation, measure_correlation, time_measure_correlation. See above comments.

        Returns
        -------
        None
        """
        # identity matrix 
        if (error_option==CovarianceStructure.identity) or (error_option==CovarianceStructure.variance):
            # if None, it means identity matrix 

            # if not None, need to check shape
            if error_cov is not None: 
                if len(error_cov)!=self.total_num_time:
                    raise ValueError("error_cov must have the same length as total_num_time. Expect length:" + str(self.total_num_time))

        elif error_option == CovarianceStructure.time_correlation: 
            # check the first dimension (length of DCMs)
            if len(error_cov)!=self.n_total_measurements:
                raise ValueError("error_cov must have the same length as n_total_measurements. Expect length:"+str(self.n_total_measurements))
            
            # check the time correlation matrice shape for each DCM
            # loop over the index of DCM to retrieve the number of time points for DCM
            for i in range(self.n_total_measurements):
                # check row number
                if len(error_cov[0])!=self.Nt[i]:
                    raise ValueError("error_cov[i] must have the shape Nt[i]*Nt[i]. Expect number of rows:"+str(self.Nt[i]))
                # check column number
                if len(error_cov[0][0])!=self.Nt[i]:
                    raise ValueError("error_cov[i] must have the shape Nt[i]*Nt[i]. Expect number of columns:"+str(self.Nt[i]))

        elif error_option == CovarianceStructure.measure_correlation:
            # check row number
            if len(error_cov)!=self.n_total_measurements:
                raise ValueError("error_cov must have the same length as n_total_measurements. Expect number of rows:"+str(self.n_total_measurements))
            # check column number
            if len(error_cov[0])!=self.n_total_measurements:
                raise ValueError("error_cov[i] must have the same length as n_total_measurements. Expect number of columns:"+str(self.n_total_measurements))
     
        elif error_option == CovarianceStructure.time_measure_correlation:
            # check row number
            if len(error_cov)!=self.total_num_time:
                raise ValueError("error_cov must have the shape total_num_time*total_num_time. Expect number of rows:"+str(self.total_num_time))
            # check column number
            if len(error_cov[0])!=self.total_num_time:
                raise ValueError("error_cov must have the shape total_num_time*total_num_time. Expect number of columns:"+str(self.total_num_time))

    def _dynamic_flatten(self, Q):
        """Update dynamic flattened matrix index. 
        Arguments 
        ---------
        :param Q: jacobian information for main class use, a 2D array of shape [N_total_meausrements * Np]

        Returns
        -------
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
        # loop over total measurement index
        for i in range(self.n_total_measurements):
            if i in self.static_measurement_idx: # static measurements are not flattened for dynamic flatten
                # dynamic_flatten
                Q_dynamic_flatten.append(Q[i])
                # map position index in Q_dynamic_flatten where each measurement starts
                self.head_pos_dynamic_flatten[i] = count1 
                # store all static measurements index after dynamic_flattening
                self.static_idx_dynamic_flatten.append(count1)
                self.dynamic_to_flatten[count1] = [] # static measurement's dynamic_flatten index corresponds to a list of flattened index

                # flatten 
                for t in range(len(Q[i])):
                    Q_flatten.append(Q[i][t])
                    if t==0:
                        self.head_pos_flatten[i] = count2
                    # all static measurements index after flatten
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

        Arguments
        ---------
        :param error_cov: if error_cov is None, return an identity matrix 
        option 1: a numpy vector, each element is the corresponding variance, a.k.a. diagonal elements.
            Shape: Sum(Nt) 
        option 2: a 3D numpy array, each element is the error covariances
            This option assumes covariances not between measurements, but between timepoints for one measurement
            Shape: Nm * (Nt_m * Nt_m)
        option 3: a 2D numpy array, covariance matrix for a single time steps 
            This option assumes the covariances between measurements at the same timestep in a time-invariant way 
            Shape: Nm * Nm
        option 4: a 2D numpy array, covariance matrix for the flattened measurements 
            Shape: sum(Nt) * sum(Nt) 
        :param: error_opt: CovarianceStructure
            can choose from identity, variance, time_correlation, measure_correlation, time_measure_correlation. See above comments.

        Returns
        -------
        Sigma: a 2D numpy array, covariance matrix for the flattened measurements 
            Shape: sum(Nt) * sum(Nt) 
        """
        
        # initialize error covariance matrix, shape N_all_t * N_all_t
        Sigma = np.zeros((self.total_num_time, self.total_num_time))

        # identity matrix or only have variance 
        if (error_option==CovarianceStructure.identity) or (error_option==CovarianceStructure.variance):
            # if given None, it means it is an identity matrix 
            if not error_cov:
                # create identity matrix 
                error_cov = [1]*self.total_num_time
            # loop over diagonal elements and change
            for i in range(self.total_num_time):
                # Sigma has 0 in all off-diagonal elements, error_cov gives the diagonal elements
                Sigma[i,i] = error_cov[i]

        # different time correlation matrix for each measurement 
        # no covariance between measurements
        elif error_option == CovarianceStructure.time_correlation: 
            for i in range(self.n_total_measurements):
                # give the error covariance to Sigma 
                # each measurement has a different time-correlation structure 
                # that is why this is a 3D matrix
                sigma_i_start = self.head_pos_flatten[i]
                # loop over all timepoints for measurement i 
                # for each measurement, the time correlation matrix is Nt*Nt
                for t1 in range(self.Nt[i]):
                    for t2 in range(self.Nt[i]):
                        # for the ith measurement, the error matrix is error_cov[i]
                        Sigma[sigma_i_start+t1, sigma_i_start+t2] = error_cov[i][t1][t2]

        # covariance between measurements 
        # the covariances between measurements at the same timestep in a time-invariant way
        elif error_option == CovarianceStructure.measure_correlation:
            # loop over number of measurements
            for i in range(self.n_total_measurements):
                # loop over number of measurements
                for j in range(self.n_total_measurements):
                    # find the covariance term
                    cov_ij = error_cov[i][j]
                    # find the starting index for each measurement (each measurement i has Nt[i] entries)
                    head_i = self.head_pos_flatten[i]
                    # starting index for measurement j
                    head_j = self.head_pos_flatten[j]
                    # i, j may have different timesteps
                    # we find the corresponding index by locating the starting indices
                    for t in range(min(self.Nt[i], self.Nt[j])):
                        Sigma[t+head_i, t+head_j] = cov_ij 
     
        # the full covariance matrix is given
        elif error_option == CovarianceStructure.time_measure_correlation:
            Sigma = np.asarray(error_cov)

        self.Sigma = Sigma

        return Sigma
        

    def _split_sigma(self, Sigma):
        """Split the error covariance matrix to be used for computation
        They are split to DCM-DCM (scalar) covariance, DCM-SCM (vector) covariance, SCCM-SCM (matrix) covariance 
        We inverse the Sigma for the computation of FIM

        Arguments
        ---------
        :param Sigma: a 2D numpy array, covariance matrix for the flattened measurements 
            Shape: sum(Nt) * sum(Nt)  

        Returns
        -------
        None
        """
        # Inverse of covariance matrix is used 
        # pinv is used to avoid ill-conditioning issues
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
        unit FIMs include DCM-DCM FIM, DCM-SCM FIM, SCM-SCM FIM
        """

        self.fim_collection = []

        # loop over measurement index
        for i in range(self.num_measure_dynamic_flatten):
            # loop over measurement index 
            for j in range(self.num_measure_dynamic_flatten):

                # static*static 
                if i in self.static_idx_dynamic_flatten and j in self.static_idx_dynamic_flatten:
                    #print("static * static, cov:", self.Sigma_inv[(i,j)])
                    unit = np.asarray(self.Q_dynamic_flatten[i]).T@self.Sigma_inv[(i,j)]@np.asarray(self.Q_dynamic_flatten[j])
                    
                # consider both i=SCM, j=DCM scenario and i=DCM, j=SCM scenario
                # static*dynamic
                elif i in self.static_idx_dynamic_flatten and j in self.dynamic_idx_dynamic_flatten:
                    #print("static*dynamic, cov:", self.Sigma_inv[(i,j)])
                    unit = np.asarray(self.Q_dynamic_flatten[i]).T@self.Sigma_inv[(i,j)]@np.asarray(self.Q_dynamic_flatten[j]).reshape(1,self.n_parameters)

                # static*dynamic
                elif i in self.dynamic_idx_dynamic_flatten and j in self.static_idx_dynamic_flatten:
                    #print("static*dynamic, cov:", self.Sigma_inv[(i,j)])
                    unit = np.asarray(self.Q_dynamic_flatten[i]).reshape(1, self.n_parameters).T@self.Sigma_inv[(i,j)].T@np.asarray(self.Q_dynamic_flatten[j])

                # dynamic*dynamic
                else:
                    #print("dynamic*dynamic, cov:", self.Sigma_inv[(i,j)])
                    unit = self.Sigma_inv[(i,j)]*np.asarray(self.Q_dynamic_flatten[i]).reshape(1, self.n_parameters).T@np.asarray(self.Q_dynamic_flatten[j]).reshape(1,self.n_parameters)

                # store unit FIM following this order
                self.fim_collection.append(unit.tolist())

    def __measure_matrix(self, measurement_vector):
        """
        This is a helper function, when giving a vector of solutions, it converts this vector into a 2D array
        This is needed for proofing the solutions after the optimization, 
        since we only computes the half diagonal of the measurement matrice and flatten it. 

        Arguments
        ---------
        :param measurement_vector: a vector of measurement weights solution
        
        Returns
        -------
        measurement_matrix: a full measurement matrix, construct the weights for covariances
        """
        # check if measurement vector legal
        assert len(measurement_vector)==self.total_no_measure, "Measurement vector is of wrong shape!!!"

        # initialize measurement matrix as a 2D array
        measurement_matrix = np.zeros((self.total_no_measure, self.total_no_measure))

        # loop over total measurement index 
        for i in range(self.total_no_measure):
            for j in range(self.total_no_measure):
                measurement_matrix[i,j] = min(measurement_vector[i], measurement_vector[j])

        return measurement_matrix

    def __print_FIM(self, FIM):
        """
        Analyze one given FIM, this is a helper function after the optimization. 

        Arguments
        ---------
        :param FIM: FIM matrix

        Returns
        -------
        None
        """

        det = np.linalg.det(FIM) # D-optimality
        trace = np.trace(FIM) # A-optimality 
        eig = np.linalg.eigvals(FIM)
        print('======FIM result======')
        print('FIM:', FIM)
        print('Determinant:', det, '; log_e(det):', np.log(det),  '; log_10(det):', np.log10(det))
        print('Trace:', trace, '; log_e(trace):', np.log(trace), '; log_10(trace):', np.log10(trace))
        print('Min eig:', min(eig), '; log_e(min_eig):', np.log(min(eig)), '; log_10(min_eig):', np.log10(min(eig)))
        print('Cond:', max(eig)/min(eig))

    def continuous_optimization(self, mixed_integer=False, obj=ObjectiveLib.A, 
                                mix_obj = False, alpha=1, fixed_nlp=False,
                                fix=False, upper_diagonal_only=False,
                                num_dynamic_t_name = None, 
                                manual_number=20, budget=100, 
                                init_cov_y=None, initial_fim=None,
                                dynamic_install_initial = None,
                                total_measure_initial = 1, 
                                static_dynamic_pair=None,
                                time_interval_all_dynamic=False, 
                                total_manual_num_init=10, 
                                cost_initial = 100, 
                               FIM_diagonal_small_element=0, 
                               print_level = 0):
        
        """Continuous optimization problem formulation. 

        Arguments
        ---------
        :param mixed_integer: boolean 
            not relaxing integer decisions
        :param obj: Enum
            "A" or "D" optimality, use trace or determinant of FIM 
        :param mix_obj: boolean 
            if True, the objective function is a weighted sum of A- and D-optimality (trace and determinant)
        :param alpha: float
            range [0,1], weight of mix_obj. if 1, it is A-optimality. if 0, it is D-optimality 
        :param fixed_nlp: boolean 
            if True, the problem is formulated as a fixed NLP 
        :param fix: boolean
            if solving as a square problem or with DOFs 
        :param upper_diagonal_only: boolean
            if using upper_diagonal_only set to define decisions and FIM, or not 
        :param num_dynamic_t_name: list
            a list of the exact time points for the dynamic-cost measurements time points 
        :param manual_number: integer 
            the maximum number of human measurements for dynamic measurements
        :param budget: integer
            total budget
        :param init_cov_y: list of lists
            initialize decision variables 
        :param initial_fim: list of lists
            initialize FIM
        :param dynamic_install_initial: list
            initialize if_dynamic_install
        :param total_measure_initial: integer
            initialize the total number of measurements chosen
        :param static_dynamic_pair: list of lists
            a list of the name of measurements, that are selected as either dynamic or static measurements.
        :param time_interval_all_dynamic: boolean
            if True, the minimal time interval applies for all dynamical measurements 
        :param total_manual_num_init: integer
            initialize the total number of dynamical timepoints selected 
        :param cost initial: float
            initialize the cost 
        :param FIM_diagonal_small_element: float
            a small number, default to be 0, to be added to FIM diagonal elements for better convergence
        :param print_level: integer
            0 (default): no process information 
            1: minimal info
            2: intermediate 
            3: everything

        Returns
        -------
        None
        """

        m = pyo.ConcreteModel()

        # measurements set
        m.n_responses = pyo.Set(initialize=range(self.num_measure_dynamic_flatten))
        m.num_measure_dynamic_flatten = self.num_measure_dynamic_flatten
        m.n_static_measurements = self.n_static_measurements
        m.num_measure_dynamic_flatten = self.num_measure_dynamic_flatten
        m.cost_list = self.cost_list
        m.dynamic_install_cost = self.dynamic_install_cost
        # FIM set 
        m.DimFIM = pyo.Set(initialize=range(self.n_parameters))

        self.print_level = print_level
        
        self.fixed_nlp= fixed_nlp
        self.initial_fim = initial_fim
        # dynamic measurements parameters 
        # dynamic measurement number of timepoints 
        self.dynamic_Nt = self.Nt[self.n_static_measurements]
        # dynamic measurement index number 
        # Pyomo model explicitly numbers all of the static measurements first and then all of the dynmaic measurements
        m.DimDynamic = pyo.Set(initialize=range(self.n_static_measurements, self.n_total_measurements))
        # turn dynamic measurement number of timepoints into a pyomo set 
        m.DimDynamic_t = pyo.Set(initialize=range(self.dynamic_Nt)) 
        
        # pair time index and real time 
        # for e.g., time 2h is the index 16, dynamic[16] = 2
        # this is for the time interval between two DCMs computation
        dynamic_time = {}
        # loop over time index 
        for i in range(self.dynamic_Nt):
            # index: real time
            dynamic_time[i] = num_dynamic_t_name[i]
        self.dynamic_time = dynamic_time

        # initialize with identity
        def identity(m,a,b):
            return 1 if a==b else 0
        def initialize_point(m,a,b):
            if init_cov_y[a][b] > 0:
                return init_cov_y[a][b]
            else:
                # this is to avoid that some times the given solution contains a really small negative number
                return 0
        
        if init_cov_y is not None:
            initialize=initialize_point
        else:
            initialize=identity

        # only define the upper triangle of symmetric matrices 
        def n_responses_half_init(m):
            return ((a,b) for a in m.n_responses for b in range(a, self.num_measure_dynamic_flatten))
        
        # only define the upper triangle of FIM 
        def DimFIMhalf_init(m):
            return ((a,b) for a in m.DimFIM for b in range(a, self.n_parameters))
        
        # set for measurement y matrix
        m.responses_upper_diagonal = pyo.Set(dimen=2, initialize=n_responses_half_init)
        # set for FIM 
        m.DimFIM_half = pyo.Set(dimen=2, initialize=DimFIMhalf_init)
        
        # if decision variables y are binary
        if mixed_integer:
            # if only defining upper triangle of the y matrix
            if upper_diagonal_only:
                m.cov_y = pyo.Var(m.responses_upper_diagonal, initialize=initialize, within=pyo.Binary)
            # else, define all elements in the y symmetry matrix
            else:
                m.cov_y = pyo.Var(m.n_responses, m.n_responses, initialize=initialize, within=pyo.Binary)
        # if decision variables y are relaxed
        else:
            # if only defining upper triangle of the y matrix
            if upper_diagonal_only:
                m.cov_y = pyo.Var(m.responses_upper_diagonal, initialize=initialize, bounds=(0,1), within=pyo.NonNegativeReals)
            # else, define all elements in the y symmetry matrix
            else:
                m.cov_y = pyo.Var(m.n_responses, m.n_responses, initialize=initialize, bounds=(0,1), within=pyo.NonNegativeReals)
        
        # use a fix option to compute results for square problems with given y 
        if fix or fixed_nlp:
            m.cov_y.fix()

        def init_fim(m,p,q):
            return initial_fim[p,q]
        
        if initial_fim is not None:
            # Initialize dictionary for grey-box model
            fim_initial_dict = {}
            for i in range(self.n_parameters):
                for j in range(i, self.n_parameters):
                    str_name = 'ele_'+str(i)+"_"+str(j)
                    fim_initial_dict[str_name] = initial_fim[i,j] 
        
        if upper_diagonal_only:
            m.TotalFIM = pyo.Var(m.DimFIM_half, initialize=init_fim)
        else:
            m.TotalFIM = pyo.Var(m.DimFIM, m.DimFIM, initialize=init_fim)

        ### compute FIM 
        def eval_fim(m, a, b):
            """
            Evaluate fim 
            FIM = sum(cov_y[i,j]*unit FIM[i,j]) for all i, j in n_responses

            a, b: dimensions for FIM, iterate in parameter set 
            """
            if a <= b: 
                summi = 0 
                for i in m.n_responses:
                    for j in m.n_responses:
                        # large_idx, small_idx are needed because cov_y is only defined the upper triangle matrix 
                        # the FIM order is i*num_measurement + j no matter if i is the smaller one or the bigger one
                        large_idx = max(i,j)
                        small_idx = min(i,j)
                        summi += m.cov_y[small_idx,large_idx]*self.fim_collection[i*self.num_measure_dynamic_flatten+j][a][b]   

                # if diagonal elements, a small element can be added to avoid rank deficiency
                if a==b:
                    return m.TotalFIM[a,b] == summi + FIM_diagonal_small_element
                # if not diagonal, no need to add small number 
                else:
                    return m.TotalFIM[a,b] == summi
            # FIM is symmetric so no need to compute again
            else:
                return m.TotalFIM[a,b] == m.TotalFIM[b,a]
            
        def integer_cut_0(m):
            """Compute the total number of measurements and time points selected
            This is for the inequality constraint of integer cut.
            """
            return m.total_number_measurements == sum(m.cov_y[i,i] for i in range(self.num_measure_dynamic_flatten))
        
        def integer_cut_0_ineq(m):
            """Ensure that at least one measurement or time point is selected
            integer cut that cuts the solution of all 0  
            """
            return m.total_number_measurements >=1 
    
        def total_dynamic(m):
            """compute the total number of time points from DCMs are selected 
            This is for the inequality constraint of total number of time points from DCMs < total number of measurements limit
            """
            return m.total_number_dynamic_measurements==sum(m.cov_y[i,i] for i in range(self.n_static_measurements, self.num_measure_dynamic_flatten))
            
        ### cov_y constraints
        def y_covy1(m, a, b):
            """
            cov_y[a,b] indicates if measurement a, b are both selected, i.e. a & b
            cov_y[a,b] = cov_y[a,a]*cov_y[b,b]. Relax this equation to get cov_y[a,b] <= cov_y[a,a]
            """ 
            if b > a:
                return m.cov_y[a, b] <= m.cov_y[a, a]
            else:
                # skip lower triangle constraints since y is a symmetric matrix
                return pyo.Constraint.Skip
            
        def y_covy2(m, a, b):
            """
            cov_y[a,b] indicates if measurement a, b are both selected, i.e. a & b
            cov_y[a,b] = cov_y[a,a]*cov_y[b,b]. Relax this equation to get cov_y[a,b] <= cov_y[b,b]
            """ 
            if b > a:
                return m.cov_y[a, b] <= m.cov_y[b, b]
            else:
                # skip lower triangle constraints since y is a symmetric matrix 
                return pyo.Constraint.Skip
            
        def y_covy3(m, a, b):
            """
            cov_y[a,b] indicates if measurement a, b are both selected, i.e. a & b
            cov_y[a,b] = cov_y[a,a]*cov_y[b,b]. Relax this equation to get cov_y[a,b] >= cov_y[a,a]+cov_y[b,b]-1
            """ 
            if b > a:
                return m.cov_y[a, b] >= m.cov_y[a, a] + m.cov_y[b, b] - 1
            else:
                # skip lower triangle constraints since y is a symmetric matrix 
                return pyo.Constraint.Skip
            
        def symmetry(m,a,b):
            """
            Ensure the symmetry of y matrix.
            This is only used when all elements of y are defined, do not need to be used when defining only upper triangle of y
            """
            if a<b:
                return m.cov_y[a,b] == m.cov_y[b,a]
            else:
                # skip lower triangle constraints since y is a symmetric matrix 
                return pyo.Constraint.Skip

        ### cost constraints
        def cost_compute(m):
            """Compute cost
            cost = static-cost measurement cost + dynamic-cost measurement installation cost + dynamic-cost meausrement timepoint cost 
            """
            static_and_dynamic_cost = sum(m.cov_y[i,i]*self.cost_list[i] for i in m.n_responses)
            dynamic_fixed_cost = sum(m.if_install_dynamic[j]*self.dynamic_install_cost[j-self.n_static_measurements] for j in m.DimDynamic)
            return m.cost == static_and_dynamic_cost + dynamic_fixed_cost
        
        def cost_limit(m):
            """Total cost smaller than the given budget
            """
            return m.cost <= budget 

        def total_dynamic_con(m):
            """total number of manual dynamical measurements number
            """
            return m.total_number_dynamic_measurements<=manual_number
        
        def dynamic_fix_yd(m,i,j):
            """if the install cost of one dynamical measurements should be considered 
            If no timepoints are chosen, there is no need to include this installation cost 
            """
            # map measurement index i to its dynamic_flatten index
            start = self.n_static_measurements + (i-self.n_static_measurements)*self.dynamic_Nt+j
            return m.if_install_dynamic[i] >= m.cov_y[start,start]
        
        def dynamic_fix_yd_con2(m,i):
            """if the install cost of one dynamical measurements should be considered 
            """
            # start index is the first time point idx for this measurement
            start = self.n_static_measurements + (i-self.n_static_measurements)*self.dynamic_Nt
            # end index is the last time point idx for this measurement
            end = self.n_static_measurements + (i-self.n_static_measurements+1)*self.dynamic_Nt
            # if no any time points from this DCM is selected, its installation cost should not be included
            return m.if_install_dynamic[i] <= sum(m.cov_y[j,j] for j in range(start, end))
        
        # set up design criterion
        def compute_trace(m):
            """compute trace 
            trace = sum(diag(M))
            """
            sum_x = sum(m.TotalFIM[j,j] for j in m.DimFIM)
            return sum_x

        # add constraints depending on if the FIM is defined as half triangle or all 
        if upper_diagonal_only:
            m.total_fim_constraint = pyo.Constraint(m.DimFIM_half, rule=eval_fim)
        else:
            m.total_fim_constraint = pyo.Constraint(m.DimFIM, m.DimFIM, rule=eval_fim)
        
        # if given fixed solution, no need to add the following constraints
        if not fix: 
            # total dynamic timepoints number
            m.total_number_dynamic_measurements = pyo.Var(initialize=total_manual_num_init)
            # compute total dynamic timepoints number
            m.manual = pyo.Constraint(rule=total_dynamic)
            # total dynamic timepoints < total manual number 
            m.con_manual = pyo.Constraint(rule=total_dynamic_con)

            ## integer cuts 
            # intiialize total number of measurements selected
            m.total_number_measurements = pyo.Var(initialize=total_measure_initial)
            # compute total number of measurements selected
            m.integer_cut0 = pyo.Constraint(rule=integer_cut_0)
            # let total number of measurements selected > 0, so we cut the all 0 solution
            m.integer_cut0_in = pyo.Constraint(rule=integer_cut_0_ineq)
            
            # only when the mixed-integer problem y are defined not only upper triangle 
            # this can help the performance of MIP problems so we keep it although not use it now
            if mixed_integer and not upper_diagonal_only:
                m.sym = pyo.Constraint(m.n_responses, m.n_responses, rule=symmetry)
            
            # relaxation constraints for y[a,b] = y[a]*y[b]
            m.cov1 = pyo.Constraint(m.n_responses, m.n_responses, rule=y_covy1)
            m.cov2 = pyo.Constraint(m.n_responses, m.n_responses, rule=y_covy2)
            m.cov3 = pyo.Constraint(m.n_responses, m.n_responses, rule=y_covy3)

            # dynamic-cost measurements installaction cost 
            def dynamic_install_init(m,j):
                # if there is no installation cost
                if dynamic_install_initial is None:
                    return 0
                # if there is installation cost
                else:
                    return dynamic_install_initial[j-self.n_static_measurements]
            
            # we choose that if this is a mixed-integer problem, the dynamic installation flag is in {0,1}
            if mixed_integer:
                m.if_install_dynamic = pyo.Var(m.DimDynamic, initialize=dynamic_install_init, bounds=(0,1), within=pyo.Binary)
            # if it is a NLP problem, this flag is relaxed
            else:
                m.if_install_dynamic = pyo.Var(m.DimDynamic, initialize=dynamic_install_init, bounds=(0,1))
                
            # for solving fixed problem, we fix the dynamic installation flag
            if self.fixed_nlp:
                m.if_install_dynamic.fix()
                    
            m.dynamic_cost = pyo.Constraint(m.DimDynamic, m.DimDynamic_t, rule=dynamic_fix_yd)
            m.dynamic_con2 = pyo.Constraint(m.DimDynamic, rule=dynamic_fix_yd_con2)

            # if each manual number smaller than a given limit
            if self.each_manual_number is not None:
                # loop over dynamical measurements 
                for i in range(self.n_dynamic_measurements):
                    def dynamic_manual_num(m):
                        """the timepoints for each dynamical measurement should be smaller than a given limit 
                        """
                        start = self.n_static_measurements + i*self.dynamic_Nt # the start index of this dynamical measurement
                        end = self.n_static_measurements + (i+1)*self.dynamic_Nt # the end index of this dynamical measurement
                        cost = sum(m.cov_y[j,j] for j in range(start, end))
                        return cost <= self.each_manual_number[0]
                    
                    con_name = "con"+str(i)
                    m.add_component(con_name, pyo.Constraint(expr=dynamic_manual_num))
            
            # if some measurements can only be dynamic or static
            if static_dynamic_pair is not None: 
                # loop over the index of the static, and dynamic measurements 
                for i, pair in enumerate(static_dynamic_pair):
                    def static_dynamic_pair_con(m):
                        return m.if_install_dynamic[pair[1]]+m.cov_y[pair[0],pair[0]] <= 1
                    
                    con_name = "con_sta_dyn"+str(i)
                    m.add_component(con_name, pyo.Constraint(expr=static_dynamic_pair_con))

            # if there is minimal interval constraint
            if self.min_time_interval is not None:
                # if this constraint applies to all dynamic measurements
                if time_interval_all_dynamic: 
                    for t in range(self.dynamic_Nt):
                        # end time is an open end of the region, so another constraint needs to be added to include end_time
                        #if dynamic_time[t]+discretize_time <= end_time+0.1*discretize_time:                 
                        def discretizer(m):
                            sumi = 0
                            count = 0 
                            # get the timepoints in this interval
                            while (count+t<self.dynamic_Nt) and (dynamic_time[count+t]-dynamic_time[t])<self.min_time_interval[0]:
                                for i in m.DimDynamic:
                                    surro_idx = self.n_static_measurements + (i-self.n_static_measurements)*self.dynamic_Nt + t + count
                                    sumi += m.cov_y[surro_idx, surro_idx]
                                count += 1 

                            return sumi <= 1 

                        con_name="con_discreti_"+str(i)+str(t)
                        m.add_component(con_name, pyo.Constraint(expr=discretizer))
                # if this constraint applies to each dynamic measurements, in a local way
                else:
                    for i in m.DimDynamicf:
                        for t in range(self.dynamic_Nt):
                            # end time is an open end of the region, so another constraint needs to be added to include end_time
                            #if dynamic_time[t]+discretize_time <= end_time+0.1*discretize_time:       
                                                    
                            def discretizer(m):
                                # sumi is the summation of all measurements selected during this time interval
                                sumi = 0
                                # count helps us go through each time points in this time interval
                                count = 0 
                                # get timepoints in this interval
                                while (count+t<self.dynamic_Nt) and (dynamic_time[count+t]-dynamic_time[t])<self.min_time_interval[0]:
                                    # surro_idx gets the index of the current time point 
                                    surro_idx = self.n_static_measurements + (i-self.n_static_measurements)*self.dynamic_Nt + t + count
                                    # sum up all timepoints selections
                                    sumi += m.cov_y[surro_idx, surro_idx]
                                    count += 1 

                                return sumi <= 1 

                            con_name="con_discreti_"+str(i)+str(t)
                            m.add_component(con_name, pyo.Constraint(expr=discretizer))
                        
            # total cost
            m.cost = pyo.Var(initialize=cost_initial)
            # compute total cost
            m.cost_compute = pyo.Constraint(rule=cost_compute)
            # make total cost < budget
            m.budget_limit = pyo.Constraint(rule=cost_limit)

        # set objective 
        if obj == ObjectiveLib.A: # A-optimailty
            m.Obj = pyo.Objective(rule=compute_trace, sense=pyo.maximize)

        elif obj == ObjectiveLib.D: # D-optimality
            def _model_i(b):
                # build grey-box module
                self.build_model_external(b, fim_init=fim_initial_dict)
            m.my_block = pyo.Block(rule=_model_i)

            if self.print_level >= 2: 
                print("Pyomo creates grey-box with initial FIM:", fim_initial_dict)

            # loop over parameters
            for i in range(self.n_parameters):
                # loop over upper triangle of FIM
                for j in range(i, self.n_parameters):
                    def eq_fim(m):
                        """Make FIM in this model equal to the FIM computed by grey-box. Necessary. 
                        """
                        return m.TotalFIM[i,j] == m.my_block.egb.inputs["ele_"+str(i)+"_"+str(j)]
                    
                    con_name = "con"+str(i)+str(j)
                    m.add_component(con_name, pyo.Constraint(expr=eq_fim))
            
            # initialize log det in grey-box module. Important.
            _, m.my_block.egb.outputs['log_det'] = np.linalg.slogdet(np.asarray(initial_fim))

            if self.print_level >= 2: 
                print("Pyomo initializes grey-box output log_det as:",  np.linalg.slogdet(np.asarray(initial_fim))[1])
            
            # add objective
            # if mix_obj, we use a weighted sum of A- and D-optimality
            if mix_obj: 
                m.trace = pyo.Expression(rule=compute_trace(m))
                m.logdet = pyo.Expression(rule=m.my_block.egb.outputs['log_det'])
                # obj is a weighted sum, alpha in [0,1] is the weight of A-optimality
                # when alpha=0, it mathematically equals to D-opt, when alpha=1, A-opt
                m.Obj = pyo.Objective(expr=m.logdet+alpha*m.trace, sense=pyo.maximize)
                
            else:
                # maximize logdet obj
                m.Obj = pyo.Objective(expr=m.my_block.egb.outputs['log_det'], sense=pyo.maximize)

        return m 


    def build_model_external(self, m, fim_init=None):
        """Build the model through grey-box module 

        Arguments
        ---------
        :param m: a pyomo model 
        :param fim_init: an array to initialize the FIM value in grey-box model

        Return
        ------
        None
        """
        # use the same print_level as the pyomo model
        ex_model = LogDetModel(n_parameters=self.n_parameters, initial_fim=fim_init, print_level=self.print_level)
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)

    def compute_FIM(self, measurement_vector):
        """
        Compute a total FIM given a set of measurement choice solutions 
        This is a helper function to verify solutions; It is not involved in the optimization part. 
        Each unit FIM is computed as: 
        FIM = Q1.T@y@Sigma_inv@Q2 

        Arguments
        ---------
        :param measurement_vector: a list of the length of all measurements, each element in [0,1]
            0 indicates this measurement is not selected, 1 indicates selected
            Note: Ensure the order of this list is the same as the order of Q, i.e. [CA(t1), ..., CA(tN), CB(t1), ...]
        
        Return
        ------
        FIM: a numpy array containing the total FIM given this solution
        """
        # generate measurement matrix
        measurement_matrix = self.__measure_matrix(measurement_vector)

        # compute FIM as Np*Np
        FIM = np.zeros((self.no_param, self.no_param))

        # go over all measurement index
        for m1_rank in range(self.total_no_measure):
            # get the corresponding gradient vector for this measurement 
            # use np.matrix to keep the dimensions for the vector, otherwise numpy makes the vector 1D instead of Np*1
            Q_m1 = np.matrix(self.Q[m1_rank])
            # go over all measurement index
            for m2_rank in range(self.total_no_measure):
                # get the corresponding gradient vector for this measurement 
                Q_m2 = np.matrix(self.Q[m2_rank])
                # solution of if these two measurements are selected
                measure_matrix = np.matrix([measurement_matrix[m1_rank,m2_rank]])
                # retrieve the error covariance matrix corresponding part
                sigma = np.matrix([self.Sigma_inv[m1_rank,m2_rank]])
                # compute FIM as Q.T@y@error_cov@Q
                FIM_unit = Q_m1.T@measure_matrix@sigma@Q_m2
                FIM += FIM_unit
        # FIM read
        if self.verbose:
            self.__print_FIM(FIM)

        return FIM
    
    def solve(self, mod, mip_option=False, objective=ObjectiveLib.A, degeneracy_hunter=False):
        """
        Set up solvers

        Arguments
        ---------
        :param mod: a Pyomo model 
        :mip_option: boolean, if True, it is a mixed-integer problem, otherwise it is a relaxed problem with no integer decisions
        :objective: Enum, "A" or "D" optimality, use trace or determinant of FIM 
        :degeneracy_hunter: boolean, when set up to True, use degeneracy hunter to check infeasibility in constraints. For debugging. 
        
        Return
        ------
        mod: MINLP problem returns the solved model 
        None: other problem returns none
        """
        if self.fixed_nlp:
            solver = pyo.SolverFactory('cyipopt')
            solver.config.options['hessian_approximation'] = 'limited-memory' 
            additional_options={'max_iter':3000, 'output_file': 'console_output',
                                    'linear_solver':'mumps', 
                                    #"halt_on_ampl_error": "yes", # this option seems not working for cyipopt
                                    "bound_push": 1E-10}
            
            if degeneracy_hunter:
                additional_options={'max_iter':0, 'output_file': 'console_output',
                                 'linear_solver':'mumps',  'bound_push':1E-10}

            for k,v in additional_options.items():
                solver.config.options[k] = v
            solver.solve(mod, tee=True)
            
            if degeneracy_hunter:
                dh = DegeneracyHunter(mod, solver=solver)
        
        elif not mip_option and objective==ObjectiveLib.A:
            #solver = pyo.SolverFactory('ipopt')
            #solver.options['linear_solver'] = "ma57"
            #solver.solve(mod, tee=True)
                
            solver = pyo.SolverFactory('gurobi', solver_io="python")
            solver.options['mipgap'] = 0.1
            solver.solve(mod, tee=True)

        elif mip_option and objective==ObjectiveLib.A:
            solver = pyo.SolverFactory('gurobi', solver_io="python")
            #solver.options['mipgap'] = 0.1
            solver.solve(mod, tee=True)
            
        elif not mip_option and objective==ObjectiveLib.D:  
            solver = pyo.SolverFactory('cyipopt')
            solver.config.options['hessian_approximation'] = 'limited-memory' 
            additional_options={'max_iter':3000, 'output_file': 'console_output',
                                'linear_solver':'mumps'}
            
            if degeneracy_hunter:
                additional_options={'max_iter':0, 'output_file': 'console_output',
                                 'linear_solver':'mumps',  'bound_push':1E-6}

            for k,v in additional_options.items():
                solver.config.options[k] = v
            solver.solve(mod, tee=True)
            
            if degeneracy_hunter:
                dh = DegeneracyHunter(mod, solver=solver)

        elif mip_option and objective==ObjectiveLib.D:
            
            solver = pyo.SolverFactory("mindtpy")

            results = solver.solve(
                mod, 
                strategy="OA",  
                init_strategy = "rNLP",
                #init_strategy='initial_binary',
                mip_solver = "gurobi", 
                nlp_solver = "cyipopt", 
                calculate_dual_at_solution=True,
                tee=True,
                #add_no_good_cuts=True,
                stalling_limit=1000,
                iteration_limit=150,
                mip_solver_tee = True, 
                mip_solver_args= {
                    "options": {
                        "NumericFocus": '3'
                    }
                },
                nlp_solver_tee = True,
                nlp_solver_args = {
                    "options": {
                        "hessian_approximation": "limited-memory", 
                        'output_file': 'console_output',
                        "linear_solver": "mumps",
                        "max_iter": 3000,   
                        #"halt_on_ampl_error": "yes", 
                        "bound_push": 1E-10,
                        "warm_start_init_point": "yes",
                        "warm_start_bound_push": 1E-10,
                        "warm_start_bound_frac": 1E-10, 
                        "warm_start_slack_bound_frac": 1E-10, 
                        "warm_start_slack_bound_push": 1E-10, 
                        "warm_start_mult_bound_push": 1E-10,
                    }
                },
            )
            
        if degeneracy_hunter:
            return mod, dh
        else:
            return mod

    def continuous_optimization_cvxpy(self, objective='D', budget=100, solver=None):
        """
        This optimization problem can also be formulated and solved in the CVXPY framework. 
        This is a generalization code for CVXPY problems for reference, not currently used for the paper.
        
        Arguments
        ---------
        :param objective: can choose from 'D', 'A', 'E' for now. if defined others or None, use A-optimality.
        :param cost_budget: give a total limit for costs.
        :param solver: default to be MOSEK. Look for CVXPY document for more solver information.
        
        Returns
        -------
        None
        """

        # compute Atomic FIM
        self.fim_computation()

        # evaluate fim in the CVXPY framework
        def eval_fim(y):
            """Evaluate FIM from y solution
            FIM = sum(cov_y[i,j]*unit FIM[i,j]) for all i, j in n_responses
            """
            fim = sum(y[i,j]*self.fim_collection[i*self.total_no_measure+j] for i in range(self.total_no_measure) for j in range(self.total_no_measure))
            return fim

        def a_opt(y):
            """A-optimality as OBJ. 
            """
            fim = eval_fim(y)
            return cp.trace(fim)
            
        def d_opt(y):
            """D-optimality as OBJ
            """
            fim = eval_fim(y)
            return cp.log_det(fim)

        def e_opt(y):
            """E-optimality as OBJ
            """
            fim = eval_fim(y)
            return -cp.lambda_min(fim)

        # construct variables
        y_matrice = cp.Variable((self.total_no_measure,self.total_no_measure), nonneg=True)

        # cost limit 
        p_cons = [sum(y_matrice[i,i]*self.cost[i] for i in range(self.total_no_measure)) <= budget]

        # loop over all measurement index
        for k in range(self.total_no_measure):
            # loop over all measurement index
            for l in range(self.total_no_measure):
                # y[k,l] = y[k]*y[l] relaxation
                p_cons += [y_matrice[k,l] <= y_matrice[k,k]]
                p_cons += [y_matrice[k,l] <= y_matrice[l,l]]
                p_cons += [y_matrice[k,k] + y_matrice[l,l] -1 <= y_matrice[k,l]] 
                p_cons += [y_matrice.T == y_matrice]

        # D-optimality
        if objective == 'D':
            obj = cp.Maximize(d_opt(y_matrice))
        # E-optimality
        elif objective =='E':
            obj = cp.Maximize(e_opt(y_matrice))
        # A-optimality
        else:
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
        Extract and show solutions from a solved Pyomo model
        mod is an argument because we 

        Arguments
        --------
        mod: a solved Pyomo model 

        Return 
        ------
        ans_y: a numpy array containing the choice for all measurements 
        sol_y: a Nd*Nt numpy array, each row contains the choice for the corresponding DCM at every timepoint
        """
        # ans_y is a numpy array of the shape Nm*Nm
        ans_y = np.zeros((self.num_measure_dynamic_flatten,self.num_measure_dynamic_flatten))

        # loop over the measurement choice index
        for i in range(self.num_measure_dynamic_flatten):
            # loop over the measurement choice index
            for j in range(i, self.num_measure_dynamic_flatten):
                cov = pyo.value(mod.cov_y[i,j])
                # give value to its symmetric part
                ans_y[i,j] = ans_y[j,i] = cov 

        # round small errors to integers
        # loop over all measurement choice
        for i in range(len(ans_y)):
            # loop over all measurement choice
            for j in range(len(ans_y[0])):
                # if the value is smaller than 0.01, we round down to 0
                if ans_y[i][j] < 0.01:
                    ans_y[i][j] = int(0)
                # if it is larger than 0.99, we round up to 1 
                elif ans_y[i][j] > 0.99:
                    ans_y[i][j] = int(1)
                # else, we keep two digits after decimal
                else: 
                    ans_y[i][j] = round(ans_y[i][j], 2)

        for c in range(self.n_static_measurements):
            print(self.measure_name[c], ": ", ans_y[c,c])

        # The DCM solutions can be very sparse and contain a lot of 0 
        # so we extract it, group it by measurement 
        sol_y = np.asarray([ans_y[i,i] for i in range(self.n_static_measurements, self.num_measure_dynamic_flatten)])
        # group DCM time points 
        sol_y = np.reshape(sol_y, (self.n_dynamic_measurements, self.dynamic_Nt))
        np.around(sol_y)
        # loop over each DCM
        for r in range(len(sol_y)):
            #print(dynamic_name[r], ": ", sol_y[r])
            print(self.measure_name[r+self.n_static_measurements])
            # print the timepoints for the current DCM
            print(sol_y[r])
            #for i, t in enumerate(sol_y[r]):
            #    if t>0.5:
            #        print(self.dynamic_time[i])

        return ans_y, sol_y



                




    

