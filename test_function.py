import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import (
    MeasurementOptimizer,
    SensitivityData,
    MeasurementData,
    CovarianceStructure,
    ObjectiveLib,
)
import pickle
import time
import unittest

class TestSensitivity(unittest.TestCase):
    """Test sensitivity object by checking the shape and values of Jacobian matrix generated
    """
    def test_sens_data(self): 
        ### Read and create Jacobian object

        # create data object to pre-compute Qs
        # read jacobian from the source csv
        # Nt is the number of time points for each measurement
        # This csv file looks like this: 
        #                 A1            A2        E1            E2
        #Unnamed: 0                                                
        #1          -1.346695 -1.665335e-13  2.223380 -1.665335e-13
        # .... 
        
        Nt = 8

        jac_info = SensitivityData("./kinetics_source_data/Q_drop0.csv", Nt)
        static_measurement_index = [
            0,
            1,
            2,
        ]  # the index of CA, CB, CC in the jacobian array, considered as SCM
        dynamic_measurement_index = [
            0,
            1,
            2,
        ]  # the index of CA, CB, CC in the jacobian array, also considered as DCM
        jac_info.get_jac_list(
            static_measurement_index,  # the index of SCMs in the jacobian array
            dynamic_measurement_index,
        )  # the index of DCMs in the jacobian array
        
        # test the shape 
        # 0,0 are both SCMs, shape should be Nt*Nt which is 8*8. same to 2,2
        self.assertTrue(np.shape(jac_info.jac)==(48,4))
        self.assertTrue(jac_info.total_measure_idx-6==0)
        
        # test if values are right 
        self.assertAlmostEqual(jac_info.jac[0,0], -1.346, places=2)
        self.assertAlmostEqual(jac_info.jac[1,0], -1.039, places=2)
        self.assertAlmostEqual(jac_info.jac[26,2], 1.6379, places=2)
        self.assertAlmostEqual(jac_info.jac[47,2], -1.568, places=2)
        self.assertAlmostEqual(jac_info.jac[47,3], -6.308, places=2)


class TestMeasurementError(unittest.TestCase):
    """Test the measurement object by checking if the code throws
    error as expected when given a wrong type of input
    """
    def test_measurement_error(self):
        # number of time points for DCM
        Nt = 8

        # maximum manual measurement number for each measurement
        max_manual_num = 10
        # minimal measurement interval
        min_interval_num = 10.0
        # maximum manual measurement number for all measurements
        total_max_manual_num = 10

        # index of columns of SCM and DCM in Q
        # SCM measurements: # CA # CB # CC. This is the index of measurement column in the reformulated Jacobian
        static_ind = [0, 1, 2]
        # DCM measurements: # CA # CB # CC. This is the index of measurement column in the reformulated Jacobian
        dynamic_ind = [3, 4, 5]
        # this index is the number of SCM + nubmer of DCM, not number of DCM timepoints
        all_ind = static_ind + dynamic_ind
        num_total_measure = len(all_ind)

        # meausrement names
        all_names_strategy3 = [
            "CA.static",
            "CB.static",
            "CC.static",
            "CA.dynamic",
            "CB.dynamic",
            "CC.dynamic",
        ]

        # define static costs in $
        # CA  # CB  # CC  # CA  # CB  # CC
        static_cost = [2000, 2000, 2000, 200, 200, 200]  

        # each static-cost measure has no per-sample cost
        dynamic_cost = [0] * len(static_ind)
        # each dynamic-cost measure costs $ 400 per sample
        dynamic_cost.extend([400] * len(dynamic_ind))

        ## define MeasurementData object with wrong jac_index number
        with self.assertRaises(ValueError):
            measure_info = MeasurementData(
                all_names_strategy3,  # name string
                num_total_measure,  # jac_index: measurement index in Q
                static_cost,  # static costs
                dynamic_cost,  # dynamic costs
                min_interval_num,  # minimal time interval between two timepoints
                max_manual_num,  # maximum number of timepoints for each measurement
                total_max_manual_num,  # maximum number of timepoints for all measurement
            )
            
        ## define MeasurementData object with wrong type of static costs
        with self.assertRaises(ValueError):
            measure_info = MeasurementData(
                all_names_strategy3,  # name string
                num_total_measure,  # jac_index: measurement index in Q
                static_cost[0],  # static costs
                dynamic_cost,  # dynamic costs
                min_interval_num,  # minimal time interval between two timepoints
                max_manual_num,  # maximum number of timepoints for each measurement
                total_max_manual_num,  # maximum number of timepoints for all measurement
            )
            
        ## define MeasurementData object with wrong type of dynamic costs
        with self.assertRaises(ValueError):
            measure_info = MeasurementData(
                all_names_strategy3,  # name string
                num_total_measure,  # jac_index: measurement index in Q
                static_cost,  # static costs
                dynamic_cost[0],  # dynamic costs
                min_interval_num,  # minimal time interval between two timepoints
                max_manual_num,  # maximum number of timepoints for each measurement
                total_max_manual_num,  # maximum number of timepoints for all measurement
            )
            
        ## define MeasurementData object with wrong type of minimal time interval
        with self.assertRaises(ValueError):
            measure_info = MeasurementData(
                all_names_strategy3,  # name string
                num_total_measure,  # jac_index: measurement index in Q
                static_cost,  # static costs
                dynamic_cost,  # dynamic costs
                "10",  # minimal time interval between two timepoints
                max_manual_num,  # maximum number of timepoints for each measurement
                total_max_manual_num,  # maximum number of timepoints for all measurement
            )
        
        ## define MeasurementData object with wrong type of max # of timepoints
        with self.assertRaises(ValueError):
            measure_info = MeasurementData(
                all_names_strategy3,  # name string
                num_total_measure,  # jac_index: measurement index in Q
                static_cost,  # static costs
                dynamic_cost[0],  # dynamic costs
                min_interval_num,  # minimal time interval between two timepoints
                "10",  # maximum number of timepoints for each measurement
                total_max_manual_num,  # maximum number of timepoints for all measurement
            )
        
        ## define MeasurementData object with wrong type of max # of total timepoints
        with self.assertRaises(ValueError):
            measure_info = MeasurementData(
                all_names_strategy3,  # name string
                num_total_measure,  # jac_index: measurement index in Q
                static_cost,  # static costs
                dynamic_cost[0],  # dynamic costs
                min_interval_num,  # minimal time interval between two timepoints
                max_manual_num,  # maximum number of timepoints for each measurement
                "20",  # maximum number of timepoints for all measurement
            )
        

class TestCovariance(unittest.TestCase):
    """Test if all five types of covariances generate the error covariance matrix correctly 
    """
    def test_covariance(self):

        ### STEP 1: set up measurement cost strategy 

        # number of time points for DCM
        Nt = 8

        # maximum manual measurement number for each measurement
        max_manual_num = 10
        # minimal measurement interval
        min_interval_num = 10.0
        # maximum manual measurement number for all measurements
        total_max_manual_num = 10

        # index of columns of SCM and DCM in Q
        # SCM measurements: # CA # CB # CC. This is the index of measurement column in the reformulated Jacobian
        static_ind = [0, 1, 2]
        # DCM measurements: # CA # CB # CC. This is the index of measurement column in the reformulated Jacobian
        dynamic_ind = [3, 4, 5]
        # this index is the number of SCM + nubmer of DCM, not number of DCM timepoints
        all_ind = static_ind + dynamic_ind
        num_total_measure = len(all_ind)

        # meausrement names
        all_names_strategy3 = [
            "CA.static",
            "CB.static",
            "CC.static",
            "CA.dynamic",
            "CB.dynamic",
            "CC.dynamic",
        ]

        # define static costs in $
        # CA  # CB  # CC  # CA  # CB  # CC
        static_cost = [2000, 2000, 2000, 200, 200, 200]  

        # each static-cost measure has no per-sample cost
        dynamic_cost = [0] * len(static_ind)
        # each dynamic-cost measure costs $ 400 per sample
        dynamic_cost.extend([400] * len(dynamic_ind))

        ## define MeasurementData object
        measure_info = MeasurementData(
            all_names_strategy3,  # name string
            all_ind,  # jac_index: measurement index in Q
            static_cost,  # static costs
            dynamic_cost,  # dynamic costs
            min_interval_num,  # minimal time interval between two timepoints
            max_manual_num,  # maximum number of timepoints for each measurement
            total_max_manual_num,  # maximum number of timepoints for all measurement
        )


        ### STEP 2: Read and create Jacobian object


        # create data object to pre-compute Qs
        # read jacobian from the source csv
        # Nt is the number of time points for each measurement
        # This csv file looks like this: 
        #                 A1            A2        E1            E2
        #Unnamed: 0                                                
        #1          -1.346695 -1.665335e-13  2.223380 -1.665335e-13
        # .... 

        jac_info = SensitivityData("./kinetics_source_data/Q_drop0.csv", Nt)
        static_measurement_index = [
            0,
            1,
            2,
        ]  # the index of CA, CB, CC in the jacobian array, considered as SCM
        dynamic_measurement_index = [
            0,
            1,
            2,
        ]  # the index of CA, CB, CC in the jacobian array, also considered as DCM
        jac_info.get_jac_list(
            static_measurement_index,  # the index of SCMs in the jacobian array
            dynamic_measurement_index,
        )  # the index of DCMs in the jacobian array


        ### Test CovarianceStructure.identity (default)
        
        # use MeasurementOptimizer to pre-compute the unit FIMs
        # do not define any error covariance structure, it should defaultly be identity matrix
        calculator = MeasurementOptimizer(
            jac_info,  # SensitivityData object
            measure_info,  # MeasurementData object
            print_level=0,  # I use highest here to see more information
        )
        
        # test the shape 
        # 0,0 are both SCMs, shape should be Nt*Nt which is 8*8. same to 2,2
        self.assertTrue(np.shape(calculator.Sigma_inv[(0,0)])==(8,8))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,2)])==(8,8))
        # 2,3 are SCM - DCM combination, shape should be Nt*1 which is 8*1. same to 2, 26
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,3)])==(8,1))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,26)])==(8,1))
        
        # test if values are right 
        self.assertAlmostEqual(calculator.Sigma_inv[(2,2)][0,0], 1.0, places=2)
        self.assertAlmostEqual(calculator.Sigma_inv[(3,3)], 1.0, places=1)
        self.assertAlmostEqual(calculator.Sigma_inv[(26,26)], 1.0, places=2)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,3)][0,0], 0.0, places=3)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,26)][7,0], 0.0, places=3)
        
        
        ### Test CovarianceStructure.variance 
        
        # initialize the variance of each measurement CA, CB, CC
        error_cov_initial = [1, 4, 8, 1, 4, 8]
        error_cov = [] 
        # for each time point in each measurement, use the variance 
        for i in error_cov_initial: 
            error_cov.extend([i]*Nt)
            
        # use MeasurementOptimizer to pre-compute the unit FIMs
        calculator = MeasurementOptimizer(
            jac_info,  # SensitivityData object
            measure_info,  # MeasurementData object
            error_cov=error_cov,  # error covariance matrix
            error_opt=CovarianceStructure.variance,  # error covariance options
            print_level=0,  # I use highest here to see more information
        )
        
        # test the shape 
        # 0,0 are both SCMs, shape should be Nt*Nt which is 8*8. same to 2,2
        self.assertTrue(np.shape(calculator.Sigma_inv[(0,0)])==(8,8))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,2)])==(8,8))
        # 2,3 are SCM - DCM combination, shape should be Nt*1 which is 8*1. same to 2, 26
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,3)])==(8,1))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,26)])==(8,1))
        
        # test if values are right 
        self.assertAlmostEqual(calculator.Sigma_inv[(2,2)][0,0], 0.125, places=2)
        self.assertAlmostEqual(calculator.Sigma_inv[(3,3)], 1.0, places=1)
        self.assertAlmostEqual(calculator.Sigma_inv[(26,26)], 0.125, places=2)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,3)][0,0], 0.0, places=3)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,26)][7,0], 0.0, places=3)
        
        ### Test CovarianceStructure.time_correlation 
        
        # create the time-correlation matrix for one measurement, shape Nt*Nt
        error_cov_initial = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        # create such matrix to be a diagonal matrix 
        error_cov_one_measure = [[0]*Nt for i in range(Nt)]
        for i in range(Nt):
            error_cov_one_measure[i][i] = error_cov_initial[i]
        # for each measurement, use the same time-correlation matrix 
        error_cov = []
        for i in range(len(all_ind)):
            error_cov.append(error_cov_one_measure)
            
        # use MeasurementOptimizer to pre-compute the unit FIMs
        calculator = MeasurementOptimizer(
            jac_info,  # SensitivityData object
            measure_info,  # MeasurementData object
            error_cov=error_cov,  # error covariance matrix
            error_opt=CovarianceStructure.time_correlation,  # error covariance options
            print_level=0,  # I use highest here to see more information
        )
        
        # test the shape 
        # 0,0 are both SCMs, shape should be Nt*Nt which is 8*8. same to 2,2
        self.assertTrue(np.shape(calculator.Sigma_inv[(0,0)])==(8,8))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,2)])==(8,8))
        # 2,3 are SCM - DCM combination, shape should be Nt*1 which is 8*1. same to 2, 26
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,3)])==(8,1))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,26)])==(8,1))
        
        # test if values are right 
        self.assertAlmostEqual(calculator.Sigma_inv[(2,2)][0,0], 10.0, places=2)
        self.assertAlmostEqual(calculator.Sigma_inv[(3,3)], 10.0, places=1)
        self.assertAlmostEqual(calculator.Sigma_inv[(26,26)], 1.25, places=2)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,3)][0,0], 0.0, places=3)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,26)][7,0], 0.0, places=3)

        
        ### Test CovarianceStructure.measure_correlation 

        # error covariance matrix
        error_cov = [[1.0, 0.1, 0.1, 0.5, 0.05, 0.05], 
                     [0.1, 4.0, 0.5, 0.05, 2.0, 0.25], 
                     [0.1, 0.5, 8.0, 0.05, 0.25, 4.0], 
                     [0.5, 0.05, 0.05, 1.0, 0.1, 0.1], 
                     [0.05, 2.0, 0.25, 0.1, 4.0, 0.5], 
                     [0.05, 0.25, 4.0, 0.1, 0.5, 8.0]]


        # use MeasurementOptimizer to pre-compute the unit FIMs
        calculator = MeasurementOptimizer(
            jac_info,  # SensitivityData object
            measure_info,  # MeasurementData object
            error_cov=error_cov,  # error covariance matrix
            error_opt=CovarianceStructure.measure_correlation,  # error covariance options
            print_level=0,  # I use highest here to see more information
        )
        
        # test the shape 
        # 0,0 are both SCMs, shape should be Nt*Nt which is 8*8. same to 2,2
        self.assertTrue(np.shape(calculator.Sigma_inv[(0,0)])==(8,8))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,2)])==(8,8))
        # 2,3 are SCM - DCM combination, shape should be Nt*1 which is 8*1. same to 2, 26
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,3)])==(8,1))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,26)])==(8,1))
        
        # test if values are right 
        self.assertAlmostEqual(calculator.Sigma_inv[(2,2)][0,0], 0.168, places=2)
        self.assertAlmostEqual(calculator.Sigma_inv[(3,3)], 1.3379, places=1)
        self.assertAlmostEqual(calculator.Sigma_inv[(26,26)], 0.168, places=2)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,3)][0,0], 0.00737, places=3)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,26)][7,0], -0.08407, places=3)
        
        
        ### Test CovarianceStructure.time_measure_correlation 
        
        # error covariance matrix should be a sum(Nt)*sum(Nt) matrix
        error_cov = [[1]*calculator.total_num_time for i in range(calculator.total_num_time)]
        
        # use MeasurementOptimizer to pre-compute the unit FIMs
        calculator = MeasurementOptimizer(
            jac_info,  # SensitivityData object
            measure_info,  # MeasurementData object
            error_cov=error_cov,  # error covariance matrix
            error_opt=CovarianceStructure.time_measure_correlation,  # error covariance options
            print_level=0,  # I use highest here to see more information
        )
        
        # test the shape 
        # 0,0 are both SCMs, shape should be Nt*Nt which is 8*8. same to 2,2
        self.assertTrue(np.shape(calculator.Sigma_inv[(0,0)])==(8,8))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,2)])==(8,8))
        # 2,3 are SCM - DCM combination, shape should be Nt*1 which is 8*1. same to 2, 26
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,3)])==(8,1))
        self.assertTrue(np.shape(calculator.Sigma_inv[(2,26)])==(8,1))
        
        # test if values are right 
        self.assertAlmostEqual(calculator.Sigma_inv[(2,2)][0,0], 0.000434, places=4)
        self.assertAlmostEqual(calculator.Sigma_inv[(3,3)], 0.000434, places=4)
        self.assertAlmostEqual(calculator.Sigma_inv[(26,26)], 0.000434, places=4)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,3)][0,0], 0.000434, places=4)
        self.assertAlmostEqual(calculator.Sigma_inv[(2,26)][7,0], 0.000434, places=4)
        
        
if __name__ == '__main__':
    unittest.main()