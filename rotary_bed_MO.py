import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, DataProcess, CovarianceStructure, ObjectiveLib
import pickle 
import time


### Set up problem 
Nt =110

max_manual_num = 5 
min_interval_num = 10

static_ind = [0,1,2,3,4,5,6,7,8,9,10]
dynamic_ind = [11,12,13,14,15]
all_ind = static_ind+dynamic_ind

num_total = len(all_ind)

all_names_strategy3 = ['Ads.gas_inlet.F', 'Ads.gas_outlet.F', 'Ads.gas_outlet.T', 
             'Des.gas_inlet.F', 'Des.gas_outlet.F', 
             'Des.gas_outlet.T',  'Ads.T_g.Value(19,10)', 
             'Ads.T_g.Value(23,10)', 'Ads.T_g.Value(28,10)', # all static
            'Ads.gas_outlet.z("CO2").static', 'Des.gas_outlet.z("CO2").static', # static z 
            'Ads.gas_outlet.z("CO2").dynamic', 'Des.gas_outlet.z("CO2").dynamic', # dynamic z 
            'Ads.z("CO2",19,10)', 'Ads.z("CO2",23,10)', 'Ads.z("CO2",28,10)']


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

dynamic_cost = [0]*len(static_ind)
dynamic_cost.extend([100]*len(dynamic_ind))

max_manual = [max_manual_num]*num_total
min_time_interval = [min_interval_num]*num_total

measure_info = pd.DataFrame({
    "name": all_names_strategy3,
    "Q_index": all_ind,
        "static_cost": static_cost,
    "dynamic_cost": dynamic_cost,
    "min_time_interval": min_time_interval, 
    "max_manual_number": max_manual
})


### Calculate FIM
dataObject = DataProcess()
dataObject.read_jacobian('./RotaryBed/Q'+str(Nt)+'_scale.csv')
Q = dataObject.get_Q_list([0,1,2,4,5,6,8,9,10,3,7],
                        [3,7,11,12,13], Nt)


calculator = MeasurementOptimizer(Q, measure_info, error_opt=CovarianceStructure.identity, verbose=True)

fim_expect = calculator.fim_computation()


num_static = len(static_ind)
num_dynamic  = len(dynamic_ind)
num_total = num_static + num_dynamic*Nt


# initialize first iteration 
budget_opt = 19000

# choose what solutions to initialize with 
initial_option = "minlp_D"
#initial_option = "milp_A"


# ==== initialization strategy ==== 
if initial_option == "milp_A":
    curr_results = np.linspace(1000, 26000, 26)
    file_name_pre, file_name_end = './rotary_results/Apr17_A_mip_', ''
    file_name_pre_fim = './rotary_results/Apr17_FIM_A_mip_'

elif initial_option == "minlp_D":
    curr_results = np.linspace(1000, 26000, 26)
    file_name_pre, file_name_end = './rotary_results/Oct31_', '_d_mip'
    file_name_pre_fim =  './rotary_results/Oct31_'


curr_results = set([int(curr_results[i]) for i in range(len(curr_results))])

if budget_opt in curr_results: # use an existed initial solutioon
    curr_budget = budget_opt

else:
    # find the closest budget
    curr_min_diff = float("inf")
    curr_budget = 25000

    for i in curr_results:
        if abs(i-budget_opt) < curr_min_diff:
            curr_min_diff = abs(i-budget_opt)
            curr_budget = i

    print("using solution at", curr_budget, " too initialize")


y_init_file = file_name_pre+str(curr_budget)+file_name_end
fim_init_file = file_name_pre_fim+str(curr_budget)+file_name_end

with open(y_init_file, 'rb') as f:
    init_cov_y = pickle.load(f)

# round to 0.25 back to 0, so that they are integer-feasible 

for i in range(561):
    for j in range(561):
        if init_cov_y[i][j] > 0.99:
            init_cov_y[i][j] = int(1)
        else:
            init_cov_y[i][j] = int(0)
            
#print(init_cov_y)
total_manual_init = 0 
dynamic_install_init = [0,0,0,0,0]

for i in range(11,561):
    if init_cov_y[i][i] > 0.01:
        total_manual_init += 1 
        
        i_pos = int((i-11)/110)
        dynamic_install_init[i_pos] = 1
        
total_measure_init = sum(init_cov_y[i][i] for i in range(561))

cost_init = budget_opt


with open(fim_init_file, 'rb') as f:
    fim_prior = pickle.load(f)
    print(fim_prior)

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

static_dynamic = [[9,11], [10,12]]
time_interval_for_all = True

dynamic_time_dict = {}
for i, tim in enumerate(num_dynamic_time):
    dynamic_time_dict[i] = tim 


mod = calculator.continuous_optimization(mixed_integer=mip_option, 
                      obj=objective, 
                    mix_obj = mix_obj_option, alpha = alpha_opt,fixed_nlp = fixed_nlp_opt,
                    fix=fix_opt, 
                    upper_diagonal_only=sparse_opt, 
                    num_dynamic_t_name = num_dynamic_time, 
                    #manual_number = manual_num, 
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
print("solver time:", t2-t1)

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
