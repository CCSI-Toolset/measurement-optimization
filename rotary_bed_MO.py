import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, SensitivityData, MeasurementData, CovarianceStructure, ObjectiveLib
import pickle 
import time


# set up problem formulation 

# number of time points for DCM
Nt =110
# maximum manual measurement number for each measurement
max_manual_num = 5 
# minimal measurement interval 
min_interval_num = 10
# maximum manual measurement number for all measurements
total_max_manual_num = 20
# index of columns of SCM and DCM in Q
static_ind = [0,1,2,3,4,5,6,7,8,9,10]
dynamic_ind = [11,12,13,14,15]
# this index is the number of SCM + nubmer of DCM, not number of DCM timepoints
all_ind = static_ind+dynamic_ind
num_total_measure = len(all_ind)
# meausrement names 
all_names_strategy3 = ['Ads.gas_inlet.F', 'Ads.gas_outlet.F', 'Ads.gas_outlet.T', 
             'Des.gas_inlet.F', 'Des.gas_outlet.F', 
             'Des.gas_outlet.T',  'Ads.T_g.Value(19,10)', 
             'Ads.T_g.Value(23,10)', 'Ads.T_g.Value(28,10)', # all static
            'Ads.gas_outlet.z("CO2").static', 'Des.gas_outlet.z("CO2").static', # static z 
            'Ads.gas_outlet.z("CO2").dynamic', 'Des.gas_outlet.z("CO2").dynamic', # dynamic z 
            'Ads.z("CO2",19,10)', 'Ads.z("CO2",23,10)', 'Ads.z("CO2",28,10)']
# define error variance 
error_variance = [1, 1, 1, 
                 1, 1, 
                 1, 1, 
                 1, 1, 
                 0.01, 0.01, 
                 0.01, 0.01,
                 0.01, 0.01, 0.01]
# define error matrix
error_mat = [[0]*len(all_names_strategy3) for _ in range(len(all_names_strategy3))]

# set up variance in the diagonal elements
for _ in range(len(all_names_strategy3)):
    error_mat[_][_] = error_variance[_]

# define static cost for static-cost measures
static_cost = [1000, #ads.gas_inlet.F (0)
                1000, #ads.gas_outlet.F (1)
                500, #ads.gas_outlet.T (2)
                1000, #des.gas_inlet.F (4)
                1000, #des.gas_outlet.F (5)
                500, #des.gas_outlet.T (6)
                1000, #ads.T19 (8)
                1000, #ads.T23 (9)
                1000, #ads.T28 (10)
                7000, #ads.z 
                7000] #des.z

# define static cost (installaion) for dynamic-cost
static_cost.extend([100, 100, 500, 500, 500])
# define dynamic cost
# each static-cost measure has no per-sample cost 
dynamic_cost = [0]*len(static_ind) # SCM has no installaion costs
# each dynamic-cost measure costs $ 100 per sample
dynamic_cost.extend([100]*len(dynamic_ind)) # 100 is the cost of each time point

# define manual number maximum 
# it is extended to the same length as measurements, so it can be one column of DataFrame
max_manual = [max_manual_num]*num_total_measure
# define minimal interval time 
# it is extended to the same length as measurements, so it can be one column of DataFrame
min_time_interval = [min_interval_num]*num_total_measure

## define MeasurementData object 
measure_info = MeasurementData(
    all_names_strategy3, # name string 
    all_ind, # jac_index: measurement index in Q
    static_cost, # static costs
    dynamic_cost, # dynamic costs
    min_interval_num, # minimal time interval between two timepoints
    max_manual, # maximum number of timepoints for each measurement
    total_max_manual_num, # maximum number of timepoints for all measurement
)

# create data object to pre-compute Qs
# read jacobian from the source csv 
# Nt is the number of time points for each measurement
jac_info = SensitivityData('./RotaryBed/Q3_scale.csv', Nt)
static_measurement_index = [0,1,2,4,5,6,8,9,10,3,7] # the index of CA, CB, CC in the jacobian array, considered as SCM
dynamic_measurement_index = [3,7,11,12,13] # the index of CA, CB, CC in the jacobian array, also considered as DCM
jac_info.get_jac_list(static_measurement_index, # the index of SCMs in the jacobian array
                    dynamic_measurement_index) # the index of DCMs in the jacobian array

# use MeasurementOptimizer to pre-compute the unit FIMs
calculator = MeasurementOptimizer(jac_info, # SensitivityData object
                                  measure_info, # MeasurementData object
                                  error_cov=error_mat, # error covariance matrix 
                                  error_opt=CovarianceStructure.measure_correlation  # error covariance options 
                                  ) 

# calculate a list of unit FIMs 
calculator.fim_computation()


## MO optimization 
mip_option = True
objective = ObjectiveLib.D
fixed_nlp_opt = False
mix_obj_option = False
alpha_opt = 0.9

sparse_opt = True
fix_opt = False
small_element = 0.0001 # the small element added to the diagonal of FIM
file_store_name = "MINLP_"

num_dynamic_time = np.linspace(2,220,Nt)

static_dynamic = [[9,11], [10,12]] # These pairs cannot be chosen simultaneously
time_interval_for_all = True

# map the timepoint index to its real time
dynamic_time_dict = {}
for i, tim in enumerate(num_dynamic_time):
    dynamic_time_dict[i] = tim 

# give range of budgets for this case
budget_ranges = np.linspace(1000, 26000, 26)
# give a trial ranges for a test; we use the first 3 budgets in budget_ranges
trial_budget_ranges = budget_ranges[:3]
# initialize with A-opt. MILP solutions
# choose what solutions to initialize from: 
# minlp_D: initialize with minlp_D solutions
# milp_A: initialize with milp_A solutions
# lp_A: iniitalize with lp_A solution 
# nlp_D: initialize with nlp_D solution
initializer_option = "milp_A"

# budgets for the current results
curr_results = np.linspace(1000, 26000, 26)

# ==== initialization strategy ==== 
# according to the initializer option, we provide different sets of initialization files 
if initializer_option == "milp_A":
    # initial solution file path and name 
    file_name_pre, file_name_end = './rotary_results/Apr17_A_mip_', ''
    
elif initializer_option == "minlp_D":
    # initial solution file path and name 
    file_name_pre, file_name_end = './rotary_results/Dec7_', '_d_mip'

elif initializer_option == "lp_A":
    # initial solution file path and name 
    file_name_pre, file_name_end = './rotary_results/May12_', '_a'

elif initializer_option == "nlp_D":
    file_name_pre, file_name_end = './rotary_results/May10_', '_d'


# initialize the initial solution dict. key: budget. value: initial solution file name 
# this initialization dictionary provides a more organized input format for initialization
initial_solution = {}
# loop over budget
for b in curr_results:
    initial_solution[b] = file_name_pre + str(b) + file_name_end


# ===== run a test for a few budgets =====
    
# use a starting budget to create the model 
start_budget = trial_budget_ranges[0]
# timestamp for creating pyomo model 
t1 = time.time()
# call the optimizer function to formulate the model and solve for the first time 
# optimizer method will 1) create the model and save as self.mod 2) initialize the model 
calculator.optimizer(mixed_integer=mip_option, # if relaxing integer decisions
                    obj=objective,  # objective function options, A or D
                    mix_obj = mix_obj_option,  # if mixing A- and D-optimality to be the OF
                    alpha = alpha_opt, # the weight of A-optimality if using mixed obj
                    fixed_nlp = fixed_nlp_opt, # if it is a fixed NLP problem
                    fix=fix_opt,  # if it is a squared problem
                    upper_diagonal_only=sparse_opt, # if only defining upper triangle part for symmetric matrix
                    num_dynamic_t_name = num_dynamic_time, # number of time points of DCMs
                    static_dynamic_pair=static_dynamic, # if one measurement can be both SCM and DCM
                    time_interval_all_dynamic = time_interval_for_all, # time interval for time points of DCMs
                    FIM_diagonal_small_element=small_element, # a small element added for FIM diagonals to avoid ill-conditioning
                    print_level=1) # print level for optimization part 

# timestamp for solving pyomo model
t2 = time.time()
calculator.solve(mip_option=mip_option, objective = objective)
# timestamp for finishing 
t3 = time.time()
print("model and solver wall clock time:", t3-t1)
print("solver wall clock time:", t3-t2)
calculator.extract_store_sol(start_budget, file_store_name)

# loop over all budgets for a test
for b in trial_budget_ranges[1:]:
    print("====Solving with budget:", b, "====")
    # open the update toggle every time so no need to create model every time
    calculator.update_budget()
    # solve the model 
    calculator.solve(mip_option=mip_option, objective = objective)
    # extract and select solutions 
    calculator.extract_store_sol(b, file_store_name)


# continue to run the rest of budgets if the test goes well 
for b in budget_ranges[3:]:
    print("====Solving with budget:", b, "====")
    # open the update toggle every time so no need to create model every time
    calculator.update_budget()
    # solve the model 
    calculator.solve(mip_option=mip_option, objective = objective)
    # extract and select solutions 
    calculator.extract_store_sol(b, file_store_name)
