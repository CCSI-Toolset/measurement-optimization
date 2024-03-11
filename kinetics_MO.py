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

# error covariance matrix
error_cov = [
    [1, 0.1, 0.1, 1, 0.1, 0.1],
    [0.1, 4, 0.5, 0.1, 4, 0.5],
    [0.1, 0.5, 8, 0.1, 0.5, 8],
    [1, 0.1, 0.1, 1, 0.1, 0.1],
    [0.1, 4, 0.5, 0.1, 4, 0.5],
    [0.1, 0.5, 8, 0.1, 0.5, 8],
]

## change the correlation of DCM VS SCM to half the original value
# variance
var_list = [error_cov[i][i] for i in range(len(error_cov))]
# standard deviation
std_list = [np.sqrt(var_list[i]) for i in range(len(var_list))]

# Generate correlation matrix
corr_original = [[0] * len(var_list) for i in range(len(var_list))]

# compute correlation, corr[i,j] = cov[i,j]/std[i]/std[j]
for i in range(len(var_list)):
    # loop over Nm
    for j in range(len(var_list)):
        corr_original[i][j] = error_cov[i][j] / std_list[i] / std_list[j]

# DCM-SCM correlations are half of DCM-DCM or SCM-SCM correlation
corr_num = 0.5

# loop over SCM rows
for i in range(len(static_ind)):
    # loop over DCM columns
    for j in range(len(static_ind), len(var_list)):
        # correlation matrix is symmetric
        # we make DCM-SCM correlation half of its original value
        corr_original[i][j] *= corr_num
        corr_original[j][i] *= corr_num

# update covariance matrix
# change back from correlation matrix
for i in range(len(var_list)):
    for j in range(len(var_list)):
        error_cov[i][j] = corr_original[i][j] * std_list[i] * std_list[j]


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


### STEP 3: Create MeasurementOptimizer object, precomputation atom FIMs


# use MeasurementOptimizer to pre-compute the unit FIMs
calculator = MeasurementOptimizer(
    jac_info,  # SensitivityData object
    measure_info,  # MeasurementData object
    error_cov=error_cov,  # error covariance matrix
    error_opt=CovarianceStructure.measure_correlation,  # error covariance options
    print_level=3,  # I use highest here to see more information
)

# calculate a list of unit FIMs
calculator.assemble_unit_fims()


### STEP 4: Create and solve MO optimization framework

## The most often used options are listed as arguments here 
mip_option_value = True # mixed integer problem or not
objective_value = ObjectiveLib.D # objective, ObjectiveLib.A or ObjectiveLib.D 
small_element_value = 0.0001  # the small element added to the diagonal of FIM
file_store_name_value = "test_run_" # save result names with this string 
# initialize with A-opt. MILP solutions
# choose what solutions to initialize from:
# minlp_D: initialize with minlp_D solutions
# milp_A: initialize with milp_A solutions
# lp_A: iniitalize with lp_A solution
# nlp_D: initialize with nlp_D solution
initializer_option_value = "lp_A"
# if run all results or just sensitivity test 
rerun_all_paper_results_value = True
# choose linear solver here. Directly comment this line or give None if default linear solver is used. 
linear_solver_opt_value = "ma57"

def kinetics_experiment(mip_option, 
                        objective, 
                        small_element=0.0001, 
                        file_store_name="test", 
                        initializer_option="lp_A", 
                        rerun_all_paper_results=False, 
                        linear_solver_opt="ma57"):
    """
    This function runs the MO experiments. 

    Arguments 
    ---------
    mip_option: boolean, mixed integer problem or not
    objective: objective function, chosen from ObjectiveLib.A or ObjectiveLib.D 
    small_element: scaler, the small element added to the diagonal of FIM
    file_store_name: string, save result names with this string 
    initializer_option: string, choose what solutions to initialize from, lp_A, nlp_D, milp_A, minlp_D are options
    rerun_all_paper_results: boolean, if run all results or just sensitivity test 
    linear_solver_opt: choose linear solver here. Directly comment this line or give None if default linear solver is used.

    Return 
    ------
    None. 
    """
    # optimization options
    fixed_nlp_opt = False
    mix_obj_option = False
    alpha_opt = 0.9

    sparse_opt = True
    fix_opt = False


    num_dynamic_time = np.linspace(0, 60, 9)

    static_dynamic = [[0, 3], [1, 4], [2, 5]]  # These pairs cannot be chosen simultaneously
    time_interval_for_all = True

    # map the timepoint index to its real time
    dynamic_time_dict = {}
    for i, tim in enumerate(num_dynamic_time[1:]):
        dynamic_time_dict[i] = np.round(tim, decimals=2)

    if rerun_all_paper_results:
        # give range of budgets for this case
        budget_ranges = np.linspace(1000, 5000, 11)
    else:
        # give a trial ranges for a test; we use the first 3 budgets in budget_ranges
        budget_ranges = [2200, 3800, 4600]

    # ==== initialization strategy ====
    # according to the initializer option, we provide different sets of initialization files
    if initializer_option == "milp_A":
        # all budgets
        curr_results = np.linspace(1000, 5000, 11)
        # initial solution file path and name
        file_name_pre, file_name_end = "./kinetics_results/MILP_", "_a"

    elif initializer_option == "minlp_D":
        # all budgets
        curr_results = np.linspace(1000, 5000, 11)
        # initial solution file path and name
        file_name_pre, file_name_end = "./kinetics_results/MINLP_", "_d"

    elif initializer_option == "lp_A":
        # all budgets
        curr_results = np.linspace(1000, 5000, 41)
        # initial solution file path and name
        file_name_pre, file_name_end = "./kinetics_results/LP_", "_a"

    elif initializer_option == "nlp_D":
        # all budgets
        curr_results = np.linspace(1000, 5000, 41)
        # initial solution file path and name
        file_name_pre, file_name_end = "./kinetics_results/NLP_", "_d"

    # initialize the initial solution dict. key: budget. value: initial solution file name
    # this initialization dictionary provides a more organized input format for initialization
    initial_solution = {}
    # loop over budget
    for b in curr_results:
        initial_solution[b] = file_name_pre + str(int(b)) + file_name_end

    # ===== run a test for a few budgets =====

    # use a starting budget to create the model
    start_budget = budget_ranges[0]
    # timestamp for creating pyomo model
    t1 = time.time()
    # call the optimizer function to formulate the model and solve for the first time
    # optimizer method will 1) create the model and save as self.mod 2) initialize the model
    calculator.optimizer(
        start_budget,  # budget
        initial_solution,  # a collection of initializations
        mixed_integer=mip_option,  # if relaxing integer decisions
        obj=objective,  # objective function options, A or D
        mix_obj=mix_obj_option,  # if mixing A- and D-optimality to be the OF
        alpha=alpha_opt,  # the weight of A-optimality if using mixed obj
        fixed_nlp=fixed_nlp_opt,  # if it is a fixed NLP problem
        fix=fix_opt,  # if it is a squared problem
        upper_diagonal_only=sparse_opt,  # if only defining upper triangle part for symmetric matrix
        num_dynamic_t_name=num_dynamic_time,  # number of time points of DCMs
        static_dynamic_pair=static_dynamic,  # if one measurement can be both SCM and DCM
        time_interval_all_dynamic=time_interval_for_all,  # time interval for time points of DCMs
        fim_diagonal_small_element=small_element,  # a small element added for FIM diagonals to avoid ill-conditioning
        print_level=1,
    )  # print level for optimization part
    # timestamp for solving pyomo model
    t2 = time.time()
    calculator.solve(mip_option=mip_option, objective=objective, linear_solver=linear_solver_opt)
    # timestamp for finishing
    t3 = time.time()
    print("model and solver wall clock time:", t3 - t1)
    print("solver wall clock time:", t3 - t2)
    calculator.extract_store_sol(start_budget, file_store_name)


    # continue to run the rest of budgets if the test goes well
    for b in budget_ranges[1:]:
        print("====Solving with budget:", b, "====")
        # open the update toggle every time so no need to create model every time
        calculator.update_budget(b)
        # solve the model
        calculator.solve(mip_option=mip_option, objective=objective, linear_solver=linear_solver_opt)
        # extract and select solutions
        calculator.extract_store_sol(b, file_store_name)

kinetics_experiment(mip_option_value, 
                    objective_value, 
                    small_element=small_element_value, 
                    file_store_name=file_store_name_value, 
                    initializer_option=initializer_option_value,
                    rerun_all_paper_results=rerun_all_paper_results_value,
                    linear_solver_opt=linear_solver_opt_value)