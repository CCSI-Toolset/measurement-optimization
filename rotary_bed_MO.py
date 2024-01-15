import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, DataProcess, CovarianceStructure, ObjectiveLib
import pickle 
import time


# set up problem formulation 

# number of time points for DCM
Nt =110
# maximum manual measurement number
max_manual_num = 5 
# minimal measurement interval 
min_interval_num = 10
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

# define static cost
static_cost = [1000, #ads.gas_inlet.F (0)
    1000, #ads.gas_outlet.F (1)
     500, #ads.gas_outlet.T (2)
    1000, #des.gas_inlet.F (4)
    1000, #des.gas_outlet.F (5)
     500, #des.gas_outlet.T (6)
     1000, #ads.T19 (8)
     1000, #ads.T23 (9)
     1000, #ads.T28 (10)
    7000,
    7000] 

static_cost.extend([100, 100, 500, 500, 500])
# define dynamic cost
dynamic_cost = [0]*len(static_ind) # SCM has no installaion costs
dynamic_cost.extend([100]*len(dynamic_ind)) # 100 is the cost of each time point

# define manual number maximum 
max_manual = [max_manual_num]*num_total_measure
# define minimal interval time 
min_time_interval = [min_interval_num]*num_total_measure

measure_info = pd.DataFrame({
    "name": all_names_strategy3,  # measurement string names
    "Q_index": all_ind, # measurement index in Q
        "static_cost": static_cost,  # static costs
    "dynamic_cost": dynamic_cost,  # dynamic costs
    "min_time_interval": min_time_interval, # minimal time interval between two timepoints
    "max_manual_number": max_manual  # maximum number of timepoints 
})


### Calculate FIM
dataObject = DataProcess()
dataObject.read_jacobian('./RotaryBed/Q'+str(Nt)+'_scale.csv')
Q = dataObject.get_Q_list([0,1,2,4,5,6,8,9,10,3,7],
                        [3,7,11,12,13], Nt)

# define with the error structure
calculator = MeasurementOptimizer(Q, measure_info, error_cov = error_mat,  
                                  error_opt=CovarianceStructure.measure_correlation, verbose=True)

# calculate unit FIMs
fim_expect = calculator.fim_computation()


## MO optimization 
# extract number of SCM, DCM, and total number of measurements
num_static = len(static_ind)
num_dynamic  = len(dynamic_ind)
# this num_total is the summation of number of SCM choices, and number of timepoints in DCMs
num_total = num_static + num_dynamic*Nt


# initialize first iteration 
budget_opt = 15000

# choose what solutions to initialize with 
#initial_option = "minlp_D" # initialize with minlp_D solutions
initial_option = "milp_A" # initialize with milp_A solutions
# initial_option = "lp_A" # iniitalize with lp_A solution 
# initialize_option = "nlp_D" # initialize with nlp_D solution


# ==== initialization strategy ==== 
if initial_option == "milp_A":
    file_name_pre, file_name_end = './rotary_results/Apr17_A_mip_', ''
    file_name_pre_fim = './rotary_results/Apr17_FIM_A_mip_'

elif initial_option == "minlp_D":
    file_name_pre, file_name_end = './rotary_results/Dec7_', '_d_mip'
    file_name_pre_fim =  './rotary_results/Dec7_fim_', 'd_mip'

elif initial_option == "lp_A":
    file_name_pre, file_name_end = './rotary_results/May12_', '_a'
    file_name_pre_fim = './rotary_results/May12_fim_', "_a"

elif initial_option == "nlp_D":
    file_name_pre, file_name_end = './rotary_results/May10_', '_d'
    file_name_pre_fim =  './rotary_results/May10_fim_', '_d'

# current results is a range containing the budgets at where the problems are solved 
curr_results = np.linspace(1000, 26000, 26)
curr_results = set([int(curr_results[i]) for i in range(len(curr_results))])
## find if there has been a original solution for the current budget
if budget_opt in curr_results: # use an existed initial solutioon
    curr_budget = budget_opt

else:
    # if not, find the closest budget, and use this as the initial point
    curr_min_diff = float("inf") # current minimal budget difference 
    curr_budget = 25000 # starting point

     # find the existing budget that minimize curr_min_diff
    for i in curr_results:
        # if we found an existing budget that is closer to the given budget
        if abs(i-budget_opt) < curr_min_diff:
            curr_min_diff = abs(i-budget_opt)
            curr_budget = i

    print("using solution at", curr_budget, " too initialize")

# assign solution file names, and FIM file names
y_init_file = file_name_pre+str(curr_budget)+file_name_end
fim_init_file = file_name_pre_fim+str(curr_budget)+file_name_end

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
dynamic_install_init = [0,0,0,0,0]

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
cost_init = budget_opt

# read FIM, initialize FIM and logdet
with open(fim_init_file, 'rb') as f:
    fim_prior = pickle.load(f)
    print(fim_prior)

# initialize FIM with a small element 
for i in range(5):
    fim_prior[i][i] += 0.0001

mip_option = True
objective = ObjectiveLib.D
fixed_nlp_opt = False
mix_obj_option = False
alpha_opt = 0.9

sparse_opt = True
fix_opt = False

manual_num = 20

num_dynamic_time = np.linspace(2,220,Nt)

static_dynamic = [[9,11], [10,12]] # These pairs cannot be chosen simultaneously
time_interval_for_all = True

# map the timepoint index to its real time
dynamic_time_dict = {}
for i, tim in enumerate(num_dynamic_time):
    dynamic_time_dict[i] = tim 

t1 = time.time()
mod = calculator.continuous_optimization(mixed_integer=mip_option, 
                      obj=objective, 
                    mix_obj = mix_obj_option, alpha = alpha_opt,fixed_nlp = fixed_nlp_opt,
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
                                        FIM_diagonal_small_element=0.0001,
                                        print_level=1)

                    
t2 = time.time()
mod = calculator.solve(mod, mip_option=mip_option, objective = objective)
t3 = time.time()

print("model and solver wall clock time:", t3-t1)
print("solver wall clock time:", t3-t2)

fim_result = np.zeros((5,5))
for i in range(5):
    for j in range(i,5):
        fim_result[i,j] = fim_result[j,i] = pyo.value(mod.TotalFIM[i,j])
        
print(fim_result)  
print('trace:', np.trace(fim_result))
print('det:', np.linalg.det(fim_result))
print(np.linalg.eigvals(fim_result))

print("Pyomo OF:", pyo.value(mod.Obj))
print("Log_det:", np.log(np.linalg.det(fim_result)))

ans_y, sol_y = calculator.extract_solutions(mod)
for i in range(5):
    print(all_names_strategy3[i+11])
    for t in range(len(sol_y[0])):
        if sol_y[i][t] > 0.95:
            print(dynamic_time_dict[t])
#print('pyomo calculated cost:', pyo.value(mod.cost))

#for i in [11,12,13,14,15]:
#    print(pyo.value(mod.if_install_dynamic[i]))
    
store = True

if store:
    file = open('Dec7_'+str(budget_opt)+'_d_mip', 'wb')

    pickle.dump(ans_y, file)

    file.close()
    
    file2 = open('Dec7_fim_'+str(budget_opt)+'_d_mip', 'wb')

    pickle.dump(fim_result, file2)

    file2.close()
