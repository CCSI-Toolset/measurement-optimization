"""
Measurement optimization tool 
@University of Notre Dame
"""
from logging import warning
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
        :param error_cov: 
            If it is a list of lists of shape (Nm*Nt)*(Nm*Nt), containing the variance-covariance matrix of all measurements
            If it is a list of lists of shape Nm*Nm, it will be duplicated for every timepoint
            If it is an Nm*1 vector, these are variances, and covariances are 0. 
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

        if not error_cov:
            # construct identity matrix
            self.Sigma = np.identity(self.total_no_measure)

            
        elif len(error_cov) == self.no_measure and type(error_cov[0]) is float:

            self.Sigma = np.identity(self.total_no_measure)

            # get variance
            for i in range(self.no_measure):
                for j in range(self.no_t):
                    self.Sigma[i*self.no_t+j, i*self.no_t+j] = error_cov[i]
            
        
        elif len(error_cov)==self.total_no_measure and len(error_cov[0])==self.total_no_measure:

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

            
        elif len(error_cov)==self.no_measure and len(error_cov[0])==self.no_measure:
            
            self.Sigma = np.identity(self.total_no_measure)
            
            # get matrix
            for i in range(self.no_measure):
                for j in range(self.no_t):
                    # variance
                    self.Sigma[i*self.no_t+j, i*self.no_t+j] = error_cov[i]

        else:
            raise warning ('Wrong inputs for error covariance matrix!!!')

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

    def continuous_optimization(self, objective='D', budget=100, solver=None):
        """

        :param objective:
        :param cost_budget:
        :return:
        """

        # compute Atomic FIM
        self.__fim_computation()

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
            if self.verbose:
                print("Use A-optimality (Trace).")
            obj = cp.Maximize(a_opt(y_matrice))

        problem = cp.Problem(obj, p_cons)

        if not solver:
            problem.solve(verbose=self.verbose)
        else:
            problem.solve(solver=solver, verbose=self.verbose)

        self.__solution_analysis(y_matrice, obj.value)
            

    def __fim_computation(self):
        """
        compute a list of FIM
        """

        self.fim_collection = []

        for i in range(self.total_no_measure):
            for j in range(self.total_no_measure):
                #unit = self.Sigma_inv[i][j]*np.matrix(self.Q[i,:]).T@np.matrix(self.Q[j,:])
                unit = self.Sigma_inv[i][j]*np.matrix(self.Q[i]).T@np.matrix(self.Q[j])
                unit_list = [[0]*self.no_param for i in range(self.no_param)]

                for k in range(self.no_param):
                    for l in range(self.no_param):
                        unit_list[k][l] = unit[k,l]

                self.fim_collection.append(unit_list)

    def __solution_analysis(self, y_value, obj_value):
        """
        """

        ## deal with y solution, round
        sol = np.zeros((self.total_no_measure,self.total_no_measure))

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

        

                

    def run_grid_search(self, enumerate_indices):

        return
