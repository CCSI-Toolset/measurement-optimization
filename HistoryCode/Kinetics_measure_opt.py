#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, DataProcess, CovarianceStructure, ObjectiveLib
#import matplotlib.pyplot as plt
import pickle 


# ## Data Process

# In[2]:


Nt = 8

max_manual_num = 10
min_interval_num = 10


# In[3]:


print(np.linspace(0,60,9))




# ### API

# In[4]:


static_ind = [0,1,2]
dynamic_ind = [3,4,5]
all_ind = static_ind+dynamic_ind

num_total = len(all_ind)


# In[5]:


all_names_strategy3 = ["CA.static", "CB.static", "CC.static", 
                      "CA.dynamic", "CB.dynamic", "CC.dynamic"]


# In[6]:


static_cost = [2000, # CA
    2000, # CB
     2000, # CC
    200, # CA
    200, # CB
     200] # CC

dynamic_cost = [0]*len(static_ind)
dynamic_cost.extend([400]*len(dynamic_ind))

max_manual = [max_manual_num]*num_total
min_time_interval = [min_interval_num]*num_total


error_cov = [[1, 0.1, 0.1, 1, 0.1, 0.1],
[0.1, 4, 0.5, 0.1, 4, 0.5],
[0.1, 0.5, 8, 0.1, 0.5, 8], 
[1, 0.1, 0.1, 1, 0.1, 0.1], 
[0.1, 4, 0.5, 0.1, 4, 0.5], 
[0.1, 0.5, 8, 0.1, 0.5, 8]]
'''

error_cov = [[1, 0.1, 0.1, 0, 0, 0],
[0.1, 4, 0.5, 0, 0, 0],
[0.1, 0.5, 8, 0, 0, 0], 
[0, 0, 0, 1, 0.1, 0.1], 
[0, 0, 0, 0.1, 4, 0.5], 
[0, 0, 0, 0.1, 0.5, 8]]
'''


# In[7]:


# correlation 
print("original covariance matrix:")
print(error_cov)

# variance 
var_list = [1,4,8,1,4,8]
std_list = [np.sqrt(var_list[i]) for i in range(6)]
print(std_list)

corr_original = [[0]*6 for i in range(6)]

for i in range(6):
    for j in range(6):
        corr_original[i][j] = error_cov[i][j]/std_list[i]/std_list[j]
        
print("original correlation matrix:")
print(corr_original)


corr_num = 0.5

# option 1
for i in range(3):
    for j in range(3,6):
        corr_original[i][j] *= corr_num
        corr_original[j][i] *= corr_num 
 # option 2
#for i in range(3):
#    corr_original[i][i+3] *= 0.99 
#    corr_original[i+3][i] *= 0.99
    
print("modified correlation matrix: ")
print(corr_original)


# In[8]:


for i in range(6):
    for j in range(6):
        error_cov[i][j] = corr_original[i][j]*std_list[i]*std_list[j]
        
print("modified covariance matrix:")
print(error_cov)


# In[9]:


print(sum(static_cost)+24*400)


# In[10]:


measure_info = pd.DataFrame({
    "name": all_names_strategy3,
    "Q_index": all_ind,
        "static_cost": static_cost,
    "dynamic_cost": dynamic_cost,
    "min_time_interval": min_time_interval, 
    "max_manual_number": max_manual
})

print(measure_info)


# In[11]:


dataObject = DataProcess()
dataObject.read_jacobian('./kinetics_fim/Q_drop0.csv')
Q = dataObject.get_Q_list([0,1,2], [0,1,2], Nt)


# In[12]:


calculator = MeasurementOptimizer(Q, measure_info, error_cov=error_cov, error_opt=CovarianceStructure.measure_correlation, verbose=True)


fim_expect = calculator.fim_computation()

print(np.shape(calculator.fim_collection))


# In[13]:


print(calculator.Sigma_inv[(23,8)])
print(calculator.Sigma_inv_matrix[44][29])


# In[14]:


print(calculator.dynamic_to_flatten)
print(calculator.head_pos_flatten)



# ## Solve

# In[18]:


num_static = len(static_ind)
num_dynamic  = len(dynamic_ind)
num_total = num_static + num_dynamic*Nt


# In[19]:


with open('./kinetics_results/May2_3000_a', 'rb') as f:
    init_cov_y = pickle.load(f)
#print(init_cov_y)


# In[ ]:


#init_cov_y = np.ones((27, 27))

#init_cov_y = np.zeros((27,27))
#for i in range(3,27):
#    for j in range(3,27):
#        init_cov_y[i,j] = 1


# In[20]:


with open('./kinetics_results/May2_fim_3000_a', 'rb') as f:
    fim_prior = pickle.load(f)
    print(fim_prior)


# In[21]:


mip_option = True
objective = ObjectiveLib.D
sparse_opt = True
fix_opt = False

manual_num = 10
budget_opt = 1000

total_manual_init = 0
dynamic_install_init = [0,0,0] 

num_dynamic_time = np.linspace(0,60,9)

static_dynamic = [[0,3],[1,4],[2,5]]
time_interval_for_all = True

dynamic_time_dict = {}
for i, tim in enumerate(num_dynamic_time[1:]):
    dynamic_time_dict[i] = np.round(tim, decimals=2)
    
print(dynamic_time_dict)


# In[ ]:


mod = calculator.continuous_optimization(mixed_integer=mip_option, 
                      obj=objective, 
                    fix=fix_opt, 
                    upper_diagonal_only=sparse_opt, 
                    num_dynamic_t_name = num_dynamic_time, 
                    manual_number = manual_num, 
                    budget=budget_opt,
                    init_cov_y= init_cov_y, 
                    initial_fim = fim_prior,
                    dynamic_install_initial = dynamic_install_init, 
                    static_dynamic_pair=static_dynamic,
                    time_interval_all_dynamic = time_interval_for_all,
                    total_manual_num_init=total_manual_init)

mod = calculator.solve(mod, mip_option=mip_option, objective = objective)


# In[ ]:


fim_result = np.zeros((4,4))
for i in range(4):
    for j in range(i,4):
        fim_result[i,j] = fim_result[j,i] = pyo.value(mod.TotalFIM[i,j])
        
print(fim_result)  
print('trace:', np.trace(fim_result))
print('det:', np.linalg.det(fim_result))
print(np.linalg.eigvals(fim_result))

print("Pyomo OF:", pyo.value(mod.Obj))
print("Log_det:", np.log(np.linalg.det(fim_result)))

ans_y, sol_y = calculator.extract_solutions(mod)
print('pyomo calculated cost:', pyo.value(mod.cost))
print("if install dynamic measurements:")
print(pyo.value(mod.if_install_dynamic[3]))
print(pyo.value(mod.if_install_dynamic[4]))
print(pyo.value(mod.if_install_dynamic[5]))


# In[ ]:


store = True

if store:
    file = open('./kinetics_results/May9_'+str(budget_opt)+'_a', 'wb')

    pickle.dump(ans_y, file)

    file.close()
    
    file2 = open('./kinetics_results/May9_fim_'+str(budget_opt)+'_a', 'wb')

    pickle.dump(fim_result, file2)

    file2.close()


# In[ ]:


for i in range(3):
    print(all_names_strategy3[i+3])
    for t in range(len(sol_y[0])):
        if sol_y[i][t] > 0.95:
            print(dynamic_time_dict[t])


# In[ ]:




