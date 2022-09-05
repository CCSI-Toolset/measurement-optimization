"""
Measurement optimization tool 
@University of Notre Dame
"""
import numpy as np
import pandas as pd
import cvxpy as cp
import warnings

class MeasurementOptimizer:
    def __init__(self, Q, no_measure, no_t, cost, error_cov=None, verbose=True):
        """
        Argument
        --------
        :param Q: a list of lists containing Jacobian matrix. It should be an m*n matrix, n is No. of parameters, m is the No. of measurements times No. of timepoints.
            Note: Q should be stacked in a way where time points of one measurement are neighbours. For e.g., [CA(t1), ..., CA(tN), CB(t1), ...]
        :param no_measure: No. of measurement items
        :param no_t: No. of time points. Note: this number should be the same for all measurement items
        :param cost: A list, containing the cost of each timepoint of corresponding measurement
        :param error_cov: A list of lists, containing the variance-covariance matrix of all measurements
            If None, the default error variance-covariance matrix contains 1 for all variances, and 0 for all covariance
            i.e. a diagonal identity matrix
        :param verbose: if print debug sentences
        """
        self.Q = Q
        self.no_measure = no_measure
        self.no_t = no_t
        self.cost = cost
        self.verbose = verbose

        # check the shape of every input, make sure they are consistent
        self.__check(Q, no_measure, no_t, cost, error_cov)

        # build and check PSD of Sigma
        self.sigma = self.__build_sigma(error_cov)


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
        # check error_cov if there is any
        if error_cov is not None:
            # check shape
            assert len(error_cov) == self.total_no_measure, "Check error covariance shape!!!"
            assert len(error_cov[0]) == self.total_no_measure, "Check error covariance shape!!!"
            self.Sigma = error_cov

            # check if error covariance is PSD
            self.Sigma_array = np.asarray(self.Sigma)

            assert np.all(np.linalg.eigvals(self.Sigma_array)>0), "Error covariance matrix is not positive semi-definite."

            # warn user if Sigma is ill-conditioning, set a bar as min(eig) < 10^{-8} or cond>10^6
            if min(np.linalg.eigvals(self.Sigma_array)) < 0.00000001 or np.linalg.cond(self.Sigma_array)>100000:
                warnings.warn("Careful...Error covariance matrix is ill-conditioning, which can cause problems.")
        else:
            # construct identity matrix
            self.Sigma = np.identity(self.total_no_measure)

        self.Sigma_inv = np.linalg.pinv(self.Sigma)

    def compute_FIM(self, measurement_vector):
        """
        Compute FIM given a set of measurements

        :param measurement_vector: a list of the length of all measurements, each element in [0,1]
            0 indicates this measurement is not selected, 1 indicates selected
            Note: Ensure the order of this list is the same as the order of Q, i.e. [CA(t1), ..., CA(tN), CB(t1), ...]
        :return:
        """
        # generate measurement matrix
        measurement_matrix = self.__measure_matrix(measurement_vector)

        # compute FIM
        FIM = np.zeros((self.no_param, self.no_param))

        for m1 in range(self.no_measure):
            for m2 in range(self.no_measure):
                for t1 in range(self.no_t):
                    for t2 in range(self.no_t):
                        m1_rank = m1*self.no_t+t1
                        m2_rank = m2*self.no_t+t2
                        Q_m1 = np.matrix(self.Q[m1_rank])
                        Q_m2 = np.matrix(self.Q[m2_rank])
                        measure_matrix = np.matrix([measurement_matrix[m1_rank,m2_rank]])
                        sigma = np.matrix([self.Sigma_inv[m1_rank,m2_rank]])
                        #if self.verbose:
                        #    print('measurement 1:', m1_rank)
                        #    print('measurement 2:', m2_rank)
                        #    print('measurement 1 Q:', Q_m1)
                        #    print('measurement 2 Q:', Q_m2)
                        #    print('is this chosen? :', measure_matrix)
                        #    print('variance/covariance:', sigma)
                        FIM_unit = Q_m1.T@measure_matrix@sigma@Q_m2
                        #if self.verbose:
                        #    print('FIM unit for the two measurements: ', FIM_unit)
                        FIM += FIM_unit
        # FIM read
        if self.verbose:
            self.__print_FIM(FIM)

        return FIM

    def __measure_matrix(self, measurement_vector):
        """

        :param measurement_vector:
        :return:
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

        :param FIM:
        :return:
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

    def continuous_optimization(self, objective='D', cost_budget=100):
        """

        :param objective:
        :param cost_budget:
        :return:
        """

        # construct variables
        for idx in range():
            cp.Variable(idx, nonneg=True)

        p_cons = [cost < budget]

        #for idx in :
        #    for t in :
        #        p_cons += [j<=1]

        p_cons += []

        obj = cp.Maximize(objective())

        problem = cp.Problem(obj, p_cons)

        return

    def run_grid_search(self, enumerate_indices):

        return
