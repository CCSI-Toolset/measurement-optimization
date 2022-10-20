import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import cvxpy as cp
from measure_optimize import MeasurementOptimizer


jaco_info = pd.read_csv('./RotaryBed/Q5.csv', index_col=False)

jaco_list = np.asarray(jaco_info)

jaco = []
for i in range(len(jaco_list)):
    jaco.append(list(jaco_list[i][1:]))
    
print(np.shape(jaco))

number_measure = 14 
number_t = 5
number_total = number_measure*number_t
cost = [1]*number_total


solution1 = [1]*number_total

calculator = MeasurementOptimizer(jaco, number_measure, number_t, cost, verbose=True)

calculator.compute_FIM(solution1)

calculator.continuous_optimization(objective='D', budget=20)