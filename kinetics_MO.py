import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, DataProcess, CovarianceStructure, ObjectiveLib
#import matplotlib.pyplot as plt
import pickle 
import time

# set up problem formulation 

# number of time points for DCM
Nt = 8

# maximum manual measurement number
max_manual_num = 10
# minimal measurement interval 
min_interval_num = 10

# index of columns of SCM and DCM in Q
static_ind = [0,1,2]
dynamic_ind = [3,4,5]
all_ind = static_ind+dynamic_ind
num_total = len(all_ind) 

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

# define dynamic costs 
dynamic_cost = [0]*len(static_ind)
dynamic_cost.extend([400]*len(dynamic_ind))

# define manual number maximum 
max_manual = [max_manual_num]*num_total
# define minimal interval time 
min_time_interval = [min_interval_num]*num_total

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
std_list = [np.sqrt(var_list[i]) for i in range(6)]

corr_original = [[0]*6 for i in range(6)]

for i in range(6):
    for j in range(6):
        corr_original[i][j] = error_cov[i][j]/std_list[i]/std_list[j]

corr_num = 0.5

for i in range(3):
    for j in range(3,6):
        corr_original[i][j] *= corr_num
        corr_original[j][i] *= corr_num 

# update covariance matrix 
for i in range(6):
    for j in range(6):
        error_cov[i][j] = corr_original[i][j]*std_list[i]*std_list[j]

## define MO  object 
measure_info = pd.DataFrame({
    "name": all_names_strategy3,
    "Q_index": all_ind,
        "static_cost": static_cost,
    "dynamic_cost": dynamic_cost,
    "min_time_interval": min_time_interval, 
    "max_manual_number": max_manual
})
dataObject = DataProcess()
dataObject.read_jacobian('./kinetics_fim/Q_drop0.csv')
Q = dataObject.get_Q_list([0,1,2], [0,1,2], Nt)


calculator = MeasurementOptimizer(Q, measure_info, error_cov=error_cov, error_opt=CovarianceStructure.measure_correlation, verbose=True)

# calculate a list of unit FIMs 
fim_expect = calculator.fim_computation()


### MO optimization framework 


# extract number of SCM, DCM, and total number of measurements
num_static = len(static_ind)
num_dynamic  = len(dynamic_ind)
num_total = num_static + num_dynamic*Nt

### initialize first iteration 
budget_opt = 3000

# choose what solutions to initialize with 
#initial_option = "minlp_D"
initial_option = "milp_A"


# ==== initialization strategy ==== 
if initial_option == "milp_A":
    curr_results = np.linspace(1000, 5000, 11)
    file_name_pre, file_name_end = './kinetics_results/May2_', '_a'

elif initial_option == "minlp_D":
    curr_results = np.linspace(1000, 5000, 11)
    file_name_pre, file_name_end = './kinetics_results/Oct21_', '_d_mip'

curr_results = set([int(curr_results[i]) for i in range(len(curr_results))])


## find if there has been a original solution for the current budget
if budget_opt in curr_results: # use an existed initial solutioon
    curr_budget = budget_opt

else:
    # if not, find the closest budget
    curr_min_diff = float("inf")
    curr_budget = 5000

    for i in curr_results:
        if abs(i-budget_opt) < curr_min_diff:
            curr_min_diff = abs(i-budget_opt)
            curr_budget = i

    print("using solution at", curr_budget, " too initialize")


y_init_file = file_name_pre+str(curr_budget)+file_name_end
fim_init_file = file_name_pre+'fim_'+str(curr_budget)+file_name_end

with open(y_init_file, 'rb') as f:
    init_cov_y = pickle.load(f)

for i in range(num_total):
    for j in range(num_total):
        if init_cov_y[i][j] > 0.99:
            init_cov_y[i][j] = int(1)
        else:
            init_cov_y[i][j] = int(0)
            
total_manual_init = 0 
dynamic_install_init = [0,0,0]

# round solutions i[]
for i in range(num_static,num_total):
    if init_cov_y[i][i] > 0.01:
        total_manual_init += 1 
        
        i_pos = int((i-num_static)/Nt)
        dynamic_install_init[i_pos] = 1
        
total_measure_init = sum(init_cov_y[i][i] for i in range(num_total))
        
# initialize cost 
cost_init = sum(dynamic_install_init)*200+total_manual_init*400 + (init_cov_y[0][0]+init_cov_y[1][1]+init_cov_y[2][2])*2000

with open(fim_init_file, 'rb') as f:
    fim_prior = pickle.load(f)
    det = np.linalg.det(fim_prior)
    
# initialize FIM with a small element 
for i in range(4):
    fim_prior[i][i] += 0.0001


mip_option = True
objective = ObjectiveLib.A
fixed_nlp_opt = False
mix_obj_option = False
alpha_opt = 0.9

sparse_opt = True
fix_opt = False

manual_num = 10

num_dynamic_time = np.linspace(0,60,9)

static_dynamic = [[0,3],[1,4],[2,5]]
time_interval_for_all = True

dynamic_time_dict = {}
for i, tim in enumerate(num_dynamic_time[1:]):
    dynamic_time_dict[i] = np.round(tim, decimals=2)


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

t1 = time.time()
mod = calculator.solve(mod, mip_option=mip_option, objective = objective)
t2 = time.time()
print("solver wall clock time:", t2-t1)
    
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


store = True

if store:
    file = open('Dec7_'+str(budget_opt)+'_d_mip', 'wb')

    pickle.dump(ans_y, file)

    file.close()
    
    file2 = open('Dec7_fim_'+str(budget_opt)+'_d_mip', 'wb')

    pickle.dump(fim_result, file2)

    file2.close()