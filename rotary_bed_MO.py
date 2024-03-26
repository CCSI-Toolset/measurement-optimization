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

# Important for loading HSL solvers
import idaes

run_tests = True
run_paper_results = False

# choose linear solver here. Directly comment out this line or give None if default linear solver is used. 
linear_solver_opt_value = "ma57"
# linear_solver_opt_value = None

'''
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
rerun_all_paper_results_value = False

# choose linear solver here. Directly comment out this line or give None if default linear solver is used. 
# linear_solver_opt_value = "ma57"
linear_solver_opt_value = None
'''

def rotary_experiment(mip_option, 
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
    initializer_option: string, choose what solutions to initialize from. 'lp_A', 'nlp_D', 'milp_A', 'minlp_D' are options
    rerun_all_paper_results: boolean, if run all results or just sensitivity test 
    linear_solver_opt: choose linear solver here. Directly comment this line or give None if default linear solver is used.

    Return 
    ------
    None. 
    """

    ### STEP 0: set up options for the MO problem

    fixed_nlp_opt = False
    mix_obj_option = False
    alpha_opt = 0.9

    sparse_opt = True
    fix_opt = False

    ### STEP 1: set up measurement cost strategy 

    # number of time points for DCM
    Nt = 110
    # maximum manual measurement number for each measurement
    max_manual_num = 5
    # minimal measurement interval
    min_interval_num = 10.0
    # maximum manual measurement number for all measurements
    total_max_manual_num = 20
    # index of columns of SCM and DCM in Q
    static_ind = [
        0,  # ads.gas_inlet.F (0)
        1,  # ads.gas_outlet.F (1)
        2,  # ads.gas_outlet.T (2)
        3,  # des.gas_inlet.F (4)
        4,  # des.gas_outlet.F (5)
        5,  # des.gas_outlet.T (6)
        6,  # ads.T19 (8)
        7,  # ads.T23 (9)
        8,  # ads.T28 (10)
        9,  # ads.gas_outlet.z("CO2")
        10,  # des.gas_outlet.z("CO2")
    ]
    # ads.gas_outlet.z("CO2") # des.gas_outlet.z("CO2") # ads.z19 # ads.z23 # ads.z28
    dynamic_ind = [11, 12, 13, 14, 15]
    # this index is the number of SCM + nubmer of DCM, not number of DCM timepoints
    all_ind = static_ind + dynamic_ind
    num_total_measure = len(all_ind)
    # meausrement names
    all_names_strategy3 = [
        "Ads.gas_inlet.F",
        "Ads.gas_outlet.F",
        "Ads.gas_outlet.T",
        "Des.gas_inlet.F",
        "Des.gas_outlet.F",
        "Des.gas_outlet.T",
        "Ads.T_g.Value(19,10)",
        "Ads.T_g.Value(23,10)",
        "Ads.T_g.Value(28,10)",  # all static
        'Ads.gas_outlet.z("CO2").static',
        'Des.gas_outlet.z("CO2").static',  # static z
        'Ads.gas_outlet.z("CO2").dynamic',
        'Des.gas_outlet.z("CO2").dynamic',  # dynamic z
        'Ads.z("CO2",19,10)',
        'Ads.z("CO2",23,10)',
        'Ads.z("CO2",28,10)',
    ]

    # define static cost for static-cost measures in $ 
    static_cost = [
        1000,  # ads.gas_inlet.F (0)
        1000,  # ads.gas_outlet.F (1)
        500,  # ads.gas_outlet.T (2)
        1000,  # des.gas_inlet.F (4)
        1000,  # des.gas_outlet.F (5)
        500,  # des.gas_outlet.T (6)
        1000,  # ads.T19 (8)
        1000,  # ads.T23 (9)
        1000,  # ads.T28 (10)
        7000,  # ads.gas_outlet.z("CO2")
        7000,  # des.gas_outlet.z("CO2")
    ]

    # define static cost (installaion) for dynamic-cost in $
    # ads.gas_outlet.z("CO2") # des.gas_outlet.z("CO2") # ads.z19 # ads.z23 # ads.z28
    static_cost.extend([100, 100, 500, 500, 500])
    # define dynamic cost
    # each static-cost measure has no per-sample cost
    dynamic_cost = [0] * len(static_ind)  # SCM has no installaion costs
    # each dynamic-cost measure costs $ 100 per sample
    dynamic_cost.extend([100] * len(dynamic_ind))  # 100 is the cost of each time point

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
    # define error variance
    error_variance = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    # define error matrix
    error_mat = [[0] * len(all_names_strategy3) for _ in range(len(all_names_strategy3))]

    # set up variance in the diagonal elements
    for i in range(len(all_names_strategy3)):
        error_mat[i][i] = error_variance[i]

    # create data object to pre-compute Qs
    # read jacobian from the source csv
    # Nt is the number of time points for each measurement
    # csv contains this dataframe (showing headers here): 
    #           MTC       HTC        DH      ISO1      ISO2
    #0    -0.539637 -0.002254 -0.049863  1.935160  0.551328
    #1    -0.494095  0.020672 -0.044344  2.097855  0.471068
    # ...
    jac_info = SensitivityData("./rotary_source_data/Q110_scale.csv", Nt)
    static_measurement_index = [
        0,  # ads.gas_inlet.F (0)
        1,  # ads.gas_outlet.F (1)
        2,  # ads.gas_outlet.T (2)
        4,  # des.gas_inlet.F (4)
        5,  # des.gas_outlet.F (5)
        6,  # des.gas_outlet.T (6)
        8,  # ads.T19 (8)
        9,  # ads.T23 (9)
        10,  # ads.T28 (10)
        3,  # ads.gas_outlet.z("CO2")
        7,  # des.gas_outlet.z("CO2")
    ]  # the index of CA, CB, CC in the jacobian array, considered as SCM
    dynamic_measurement_index = [
        3,  # ads.gas_outlet.z("CO2")
        7,  # des.gas_outlet.z("CO2")
        11,  # ads.z("CO2",19,10)
        12,  # ads.z("CO2",23,10)
        13,  # ads.z("CO2",28,10)
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
        error_cov=error_mat,  # error covariance matrix
        error_opt=CovarianceStructure.measure_correlation,  # error covariance options
        print_level=3,  # I use highest here to see all information
    )

    # calculate a list of unit FIMs
    calculator.assemble_unit_fims()

    num_dynamic_time = np.linspace(2, 220, Nt)

    static_dynamic = [[9, 11], [10, 12]]  # These pairs cannot be chosen simultaneously
    time_interval_for_all = True

    # map the timepoint index to its real time
    dynamic_time_dict = {}
    for i, tim in enumerate(num_dynamic_time):
        dynamic_time_dict[i] = tim

    if rerun_all_paper_results:
        # give range of budgets for this case
        budget_ranges = np.linspace(1000, 26000, 26)
    else:
        # give a trial ranges for a test; we use the first 3 budgets in budget_ranges
        budget_ranges = [5000, 15000]


    # budgets for the current results
    curr_results = np.linspace(1000, 26000, 26)

    # ==== initialization strategy ====
    # according to the initializer option, we provide different sets of initialization files
    if initializer_option == "milp_A":
        # initial solution file path and name
        file_name_pre, file_name_end = "./rotary_results/MILP_", "_a"

    elif initializer_option == "minlp_D":
        # initial solution file path and name
        file_name_pre, file_name_end = "./rotary_results/MINLP_", "_d"

    elif initializer_option == "lp_A":
        # initial solution file path and name
        file_name_pre, file_name_end = "./rotary_results/LP_", "_a"

    elif initializer_option == "nlp_D":
        file_name_pre, file_name_end = "./rotary_results/NLP_", "_d"

    else:
        print("Warning: initializer_option =",initializer_option," is not recognized. Please check the input.")

    # initialize the initial solution dict. key: budget. value: initial solution file name
    # this initialization dictionary provides a more organized input format for initialization
    # Note: b can be float, such as 1000.0, but it is stored as 1000, so we int(b)
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

if run_tests:

    # Default values for all tests
    small_element_value = 0.0001
    file_store_name_value = "test"

    # if run all results or just sensitivity test 
    rerun_all_paper_results_value = False

    '''
    print("\nTest A: run a test with MILP, A-optimality, and LP_A initialization")
    rotary_experiment(mip_option=True, 
                    objective=ObjectiveLib.A, 
                    small_element=small_element_value, 
                    file_store_name=file_store_name_value + "_a_", 
                    initializer_option="lp_A",
                    rerun_all_paper_results=rerun_all_paper_results_value,
                    linear_solver_opt=linear_solver_opt_value)
    '''

    print("\nTest B: run a test with NLP, D-optimality, and LP_A initialization")
    rotary_experiment(mip_option=False, 
                    objective=ObjectiveLib.D, 
                    small_element=small_element_value, 
                    file_store_name=file_store_name_value + "_b_", 
                    initializer_option="lp_A",
                    rerun_all_paper_results=rerun_all_paper_results_value,
                    linear_solver_opt=linear_solver_opt_value)
    
    print("\nTest C: run a test with MINLP, D-optimality, and MILP_A initialization")
    rotary_experiment(mip_option=True, 
                    objective=ObjectiveLib.D, 
                    small_element=small_element_value, 
                    file_store_name=file_store_name_value + "_c_", 
                    initializer_option="milp_A",
                    rerun_all_paper_results=rerun_all_paper_results_value,
                    linear_solver_opt=linear_solver_opt_value)

if run_paper_results:

    # Default values for all tests
    small_element_value = 0.0001
    file_store_name_value = "paper_alex_run"

    # if run all results or just sensitivity test 
    rerun_all_paper_results_value = True
    
    print("\nPaper Run A: MILP, A-optimality, and LP_A initialization")
    rotary_experiment(mip_option=True, 
                    objective=ObjectiveLib.A, 
                    small_element=small_element_value, 
                    file_store_name=file_store_name_value + "_a_", 
                    initializer_option="lp_A",
                    rerun_all_paper_results=rerun_all_paper_results_value,
                    linear_solver_opt=linear_solver_opt_value)
                    
    print("\nPaper Run B: NLP, D-optimality, and MILP_A initialization")
    rotary_experiment(mip_option=False,
                    objective=ObjectiveLib.D, 
                    small_element=small_element_value, 
                    file_store_name=file_store_name_value + "_b_", 
                    initializer_option="milp_A",
                    rerun_all_paper_results=rerun_all_paper_results_value,
                    linear_solver_opt=linear_solver_opt_value)
    
    print("\nPaper Run 3: MINLP, D-optimality, and MILP_A initialization")
    rotary_experiment(mip_option=True, 
                    objective=ObjectiveLib.D, 
                    small_element=small_element_value, 
                    file_store_name=file_store_name_value + "_c_", 
                    initializer_option="milp_A",
                    rerun_all_paper_results=rerun_all_paper_results_value,
                    linear_solver_opt=linear_solver_opt_value)
