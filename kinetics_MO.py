import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, SensitivityData, MeasurementData, CovarianceStructure, ObjectiveLib
import pickle 
import time

# set up problem formulation 

# number of time points for DCM
Nt = 8

# maximum manual measurement number for each measurement
max_manual_num = 10
# minimal measurement interval 
min_interval_num = 10
# maximum manual measurement number for all measurements
total_max_manual_num = 10

# index of columns of SCM and DCM in Q
static_ind = [0,1,2]
dynamic_ind = [3,4,5]
# this index is the number of SCM + nubmer of DCM, not number of DCM timepoints
all_ind = static_ind+dynamic_ind
num_total_measure = len(all_ind) 

# meausrement names 
all_names_strategy3 = ["CA.static", "CB.static", "CC.static", 
                      "CA.dynamic", "CB.dynamic", "CC.dynamic"]

# define static costs 
static_cost = [2000, # CA
                2000, # CB
                2000, # CC
                200, # CA
                200, # CB
                200] # CC

# each static-cost measure has no per-sample cost 
dynamic_cost = [0]*len(static_ind)
# each dynamic-cost measure costs $ 400 per sample
dynamic_cost.extend([400]*len(dynamic_ind))

# error covariance matrix 
error_cov = [[1, 0.1, 0.1, 1, 0.1, 0.1],
[0.1, 4, 0.5, 0.1, 4, 0.5],
[0.1, 0.5, 8, 0.1, 0.5, 8], 
[1, 0.1, 0.1, 1, 0.1, 0.1], 
[0.1, 4, 0.5, 0.1, 4, 0.5], 
[0.1, 0.5, 8, 0.1, 0.5, 8]]

## change the correlation of DCM VS SCM to half the original value 
# variance 
var_list = [1,4,8,1,4,8]
# standard deviation
std_list = [np.sqrt(var_list[i]) for i in range(len(var_list))]

# Generate correlation matrix
corr_original = [[0]*len(var_list) for i in range(len(var_list))]

# compute correlation, corr[i,j] = cov[i,j]/std[i]/std[j]
for i in range(len(var_list)):
    # loop over Nm
    for j in range(len(var_list)):
        corr_original[i][j] = error_cov[i][j]/std_list[i]/std_list[j]

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
        error_cov[i][j] = corr_original[i][j]*std_list[i]*std_list[j]

## define MeasurementData object 
measure_info = MeasurementData(
    all_names_strategy3, # name string 
    all_ind, # jac_index: measurement index in Q
    static_cost, # static costs
    dynamic_cost, # dynamic costs
    min_interval_num, # minimal time interval between two timepoints
    max_manual_num, # maximum number of timepoints for each measurement
    total_max_manual_num, # maximum number of timepoints for all measurement
)

# create data object to pre-compute Qs
# read jacobian from the source csv 
# Nt is the number of time points for each measurement
jac_info = SensitivityData('./kinetics_fim/Q_drop0.csv', Nt)
jac_info.get_jac_list([0,1,2], # the index of CA, CB, CC in the jacobian array
                          [0,1,2]) # the index of CA, CB, CC in the jacobian array

# use MeasurementOptimizer to pre-compute the unit FIMs
calculator = MeasurementOptimizer(jac_info, measure_info, error_cov=error_cov, error_opt=CovarianceStructure.measure_correlation, verbose=True)

# calculate a list of unit FIMs 
fim_expect = calculator.fim_computation()

### MO optimization framework 
# extract number of SCM, DCM, and total number of measurements
num_static = len(static_ind)
num_dynamic  = len(dynamic_ind)
# this num_total is the summation of number of SCM choices, and number of timepoints in DCMs
num_total = num_static + num_dynamic*Nt

# optimization options
mip_option = True
objective = ObjectiveLib.D
fixed_nlp_opt = False
mix_obj_option = False
alpha_opt = 0.9

sparse_opt = True
fix_opt = False
small_element = 0.0001 # the small element added to the diagonal of FIM

num_dynamic_time = np.linspace(0,60,9)

static_dynamic = [[0,3],[1,4],[2,5]] # These pairs cannot be chosen simultaneously
time_interval_for_all = True

# map the timepoint index to its real time
dynamic_time_dict = {}
for i, tim in enumerate(num_dynamic_time[1:]):
    dynamic_time_dict[i] = np.round(tim, decimals=2)

def optimizer(budget_opt, initial_option, store_name=None):
    """
    Initialize, formulate, and solve the MO problem. 

    Arguments 
    ---------
    budget_opt: budget 
    initial_option: choice of using which initial file
        # choose what solutions to initialize from: 
        # minlp_D: initialize with minlp_D solutions
        # milp_A: initialize with milp_A solutions
        # lp_A: iniitalize with lp_A solution 
        # nlp_D: initialize with nlp_D solution
    store_name: if not None, store the solution and FIM in pickle file with the given name
    """

    # ==== initialization strategy ==== 
    if initial_option == "milp_A":
        curr_results = np.linspace(1000, 5000, 11)
        file_name_pre, file_name_end = './kinetics_results/MILP_', '_a'

    elif initial_option == "minlp_D":
        curr_results = np.linspace(1000, 5000, 11)
        file_name_pre, file_name_end = './kinetics_results/MINLP_', '_d_mip'

    elif initial_option == "lp_A":
        curr_results = np.linspace(1000, 5000, 41)
        file_name_pre, file_name_end = './kinetics_results/LP_', '_a'

    elif initial_option == "nlp_D":
        curr_results = np.linspace(1000, 5000, 41)
        file_name_pre, file_name_end = './kinetics_results/NLP_', '_d'


    # current results is a range containing the budgets at where the problems are solved 
    curr_results = set([int(curr_results[i]) for i in range(len(curr_results))])

    ## find if there has been a original solution for the current budget
    if budget_opt in curr_results: # use an existed initial solutioon
        curr_budget = budget_opt

    else:
        # if not, find the closest budget, and use this as the initial point
        curr_min_diff = float("inf") # current minimal budget difference 
        curr_budget = 5000 # starting point
        
        # find the existing budget that minimize curr_min_diff
        for i in curr_results:
            # if we found an existing budget that is closer to the given budget
            if abs(i-budget_opt) < curr_min_diff:
                curr_min_diff = abs(i-budget_opt)
                curr_budget = i

        print("using solution at", curr_budget, " too initialize")

    # assign solution file names, and FIM file names
    y_init_file = file_name_pre+str(curr_budget)+file_name_end
    fim_init_file = file_name_pre+'fim_'+str(curr_budget)+file_name_end

    # read y 
    with open(y_init_file, 'rb') as f:
        init_cov_y = pickle.load(f)

    # Round possible float solution to be integer 
    for i in range(num_total):
        for j in range(num_total):
            if init_cov_y[i][j] > 0.99:
                init_cov_y[i][j] = int(1)
            else:
                init_cov_y[i][j] = int(0)
                
    # initialize total manual number 
    total_manual_init = 0 
    # initialize the DCM installation flags
    dynamic_install_init = [0,0,0]

    # round solutions
    # if floating solutions, if value > 0.01, we count it as an integer decision that is 1 or positive
    for i in range(num_static,num_total):
        if init_cov_y[i][i] > 0.01:
            total_manual_init += 1 
            
            # identify which DCM this timepoint belongs to, turn the installation flag to be positive 
            i_pos = int((i-num_static)/Nt)
            dynamic_install_init[i_pos] = 1
            
    # compute total measurements number, this is for integer cut
    total_measure_init = sum(init_cov_y[i][i] for i in range(num_total))
            
    # initialize cost, this cost is calculated by the given initial solution
    cost_init = sum(dynamic_install_init)*200+total_manual_init*400 + (init_cov_y[0][0]+init_cov_y[1][1]+init_cov_y[2][2])*2000

    # read FIM, initialize FIM and logdet
    with open(fim_init_file, 'rb') as f:
        fim_prior = pickle.load(f)
        
    # initialize FIM with a small element 
    for i in range(4):
        fim_prior[i][i] += small_element

    t1 = time.time()

    
    mod = calculator.continuous_optimization(mixed_integer=mip_option, 
                        obj=objective, 
                        mix_obj = mix_obj_option, 
                        alpha = alpha_opt,
                        fixed_nlp = fixed_nlp_opt,
                        fix=fix_opt, 
                        upper_diagonal_only=sparse_opt, 
                        num_dynamic_t_name = num_dynamic_time, 
                        budget=budget_opt,
                        init_cov_y= init_cov_y,
                        initial_fim = fim_prior,
                        dynamic_install_initial = dynamic_install_init, 
                        total_measure_initial = total_measure_init, 
                        static_dynamic_pair=static_dynamic,
                        time_interval_all_dynamic = time_interval_for_all,
                        total_manual_num_init=total_manual_init,
                        cost_initial = cost_init, 
                        FIM_diagonal_small_element=small_element,
                        print_level=1)

    t2 = time.time()
    mod = calculator.solve(mod, mip_option=mip_option, objective = objective)
    t3 = time.time()

    print("model and solver wall clock time:", t3-t1)
    print("solver wall clock time:", t3-t2)
        

    fim_result = np.zeros((4,4))
    for i in range(4):
        for j in range(i,4):
            fim_result[i,j] = fim_result[j,i] = pyo.value(mod.TotalFIM[i,j])
            
    print(fim_result)  
    print('trace:', np.trace(fim_result))
    print('det:', np.linalg.det(fim_result))
    print(np.linalg.eigvals(fim_result))

    ans_y, sol_y = calculator.extract_solutions(mod)
    print('pyomo calculated cost:', pyo.value(mod.cost))
    print("if install dynamic measurements:")
    print(pyo.value(mod.if_install_dynamic[3]))

    if store_name:
        file = open(store_name+str(budget_opt), 'wb')

        pickle.dump(ans_y, file)

        file.close()
        
        file2 = open(store_name+"_fim_"+str(budget_opt), 'wb')

        pickle.dump(fim_result, file2)

        file2.close()


