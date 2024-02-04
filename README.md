# Measurement This, Not That: Optimizing the Cost and Model-Based Information Content of Measurements

Authors: Jialu Wang, Zedong Peng, Ryan Hughes, Debangsu Bhattacharyya, David E. Bernal Neira, Alexander W. Dowling 

This repository contains code and results for paper: Measure this, not that: Optimizing the cost and model-based information content of measurements 

## Installation instructions 

The following instructions assume you have anaconda installed. We suggest create an environment with the following commands to run code: 

### Step 1: create a new environment 
- create new environment, called for e.g. `measurement_optimization`, with `conda` with `Python` version 3.8

`conda create --name measurement_optimization python=3.8`

`conda activate measurement_optimization`
   
### Step 2: install `IDAES-PSE`
- this step provides `Ipopt` solver, but this solver is not necessary for reproducing paper results
- we did not test if the code can run without this step

`pip install idaes-pse` 

`idaes get-extensions`
   
### Step 3: install `Pyomo` from specified branches
- install from the following branch for a generalization version of Mindtpy:

`pip install git+https://github.com/ZedongPeng/pyomo.git@add_mindtpy_callback`

   
### Step 4: install `GurobiPy`
- this is needed only for solving mixed-integer problems

  `conda install -c gurobi gurobi`
   
### Step 5: install `CyIpopt`
- this is needed only for D-optimality problems with grey-box modules

   `conda install -c conda-forge cyipopt`

### Software versions we use for the results 

`Python`: 3.8

`IDAES-PSE`: 2.2.0

`Pyomo`: 6.7.0 dev 0

`GurobiPy`: 10.0.3

`CyIpopt`: 1.3.0

## Code 

- `measure_optimize.py`: Measurement optimization optimization framework

- `greybox_generalize.py`: Grey-box generalization 

- `kinetics_MO.py`: Kinetics case study

- `rotary_bed_MO.py`: Rotary bed case study

- `Draw_figure.ipynb`: Generates all results figures in the manuscript


## Example to run code and reproduce figures for case studies

Setup the scripts to reproduce result files and figures from the paper: 

### Kinetics case study 

- Step 1: run `kinetics_MO.py`
- Step 2: change budgets with the variable `budget_opt` in line 301.

  Suggested ranges: 

  In our results, we use the budget range [1000, 5000] with a 100 discretization,
  i.e. [1000, 1100, ..., 5000], for relaxed problems

  we use the budget range [1000, 5000] with a 400 discretization,
  i.e. [1000, 1400, 1800, ..., 5000] for mixed-integer problems.

  You can choose the same budget ranges, or only a few points of budgets, to run the script.

- Step 3: store results for drawing figures

  To do this, in line 312, define the param `store_name` with a string you given, for e.g., "MINLP_result_".

  Then both the solutions and the FIM of the results are stored separately.

  For e.g., if running in the range [1000, 5000], the stored files will be:

  MINLP_result_1000, MINLP_result_fim_1000,

  ...

  MINLP_result_5000, MINLP_result_fim_5000,
  
- Step 4: use draw_figure.ipynb to read stored FIM and solutions

  - `read_fim` receives the string name, for.e.g. `MINLP_result_`, and budget ranges, returns a list of A- and D-optimality values of the given FIMs
 
  - `plot_data` receives both the A- and D-optimality of all four optimization strategies, and draw two figures like Fig. 3 in paper
 
  - `read_solution` receives the string name, for e.g. `MINLP_result_`, and budget ranges, returns 6 lists: CA, CB, CC solutions as SCM and DCM,
    each list contains four lists as results from four strategies
 
  - `plot_one_solution` receives and draws the solution of one measurement under four strategies. To reproduce result figure like Fig. S-2 in paper, call it 6 times to draw all 6 figures and combine to a panel figure. 

  

### Rotary bed case study 

run `rotary_bed_MO.py`. This is under working, following similar patterns as running `kinetics_MO.py`. 

## Source files

### Kinetics case study 

- `./kinetics_source_data/reactor_kinetics.py`: kinetics case study model  
- `./kinetics_source_data/Q_drop0.csv`: contain Jacobian for this case study, data structure as the following:
  
  0   |  A1  |  A2  |  E1  |  E2  |
  
  1   |  num | num  | num  | num  |

  ...

  24  |  num | num  | num  | num  |

  Rows: measurements (C_A, C_B, C_C, each measurement has 8 time points)
  Columns: parameters (4 parameters)

### Rotary bed case study 

- `./rotary_source_data/RotaryBed-DataProcess.ipynb`: process rotary bed measurements data from `Aspen Custom Modeler`, generate Jacobian
- `./rotary_source_data/Q110_scale.csv`: contain Jacobian for this case study, data structure as the following:

  0   |  MTC  |  HTC  |  DH  |  ISO1  |  ISO2 |
   
  1   |  num | num  | num  | num  | num |

  ...

  1540  |  num | num  | num  | num  | num |

  Rows: measurements (14 measurements, each has 110 time points)
  Columns: parameters (5 parameters)


## Result files 

### Kinetics case study

At each budget, the FIM result and the optimal solution are stored separately in `pickle` files. 
Computational details including solver time and numbers of operations are also stored separately in `pickle` files.

#### FIM of final results 

An example name: `LP_fim_1000_a`, the results of A-optimality LP problem of a budget of 1000 

Data file type: `pickle`, storing a numpy array of FIM of the shape Np*Np, Np is the number of parameters 

To replicate the results, iterate in the given budget range to retrieve the FIM stored in each data file

- A-optimality LP results: `kinetics_results/LP_fim_x_a`, x in the range [1000, 1100, 1200, ..., 5000]

- A-optimality MILP results: `kinetics_results/MILP_fim_x_a`, x in the range [1000, 1400, 1800, ..., 5000]

- D-optimality NLP results: `kinetics_results/NLP_fim_x_d`, x in the range [1000, 1100, 1200, ..., 5000]

- D-optimality MINLP results: `kinetics_results/MINLP_fim_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

- Operating cost results: `kinetics_results/Operate_fim_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

#### Optimal solutions

An example name: `LP_1000_a`, the results of A-optimality LP problem of a budget of 1000 

Data file type: `pickle`, storing a numpy array of the solutions of the shape Nm*Nm, Nm is the number of all measurements

- A-optimality LP results: `kinetics_results/LP_x_a`, x in the range [1000, 1100, 1200, ..., 5000]

- A-optimality MILP results: `kinetics_results/MILP_x_a`, x in the range [1000, 1400, 1800, ..., 5000]

- D-optimality NLP results: `kinetics_results/NLP_x_d`, x in the range [1000, 1100, 1200, ..., 5000]

- D-optimality MINLP results: `kinetics_results/MINLP_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

- Operating cost results: `kinetics_results/Operate_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

#### Computational details 

The computational details are stored separately. 

For A-optimality LP and MILP problems, the `pickle` files store a numpy array of the solver time of each budget 

For D-optimality NLP and MINLP problems, the `pickle` files store a dictionary, where the keys are the budgets. An example is: 

nlp_time={1000: {"t": 0.01, "n": 10}, ..., "5000": {"t": 0.01, "n": 10}}

For each budget, the value is a dictionary where the key `t` stores the solver time, `n` stores the number of iterations

- A-optimality LP solver time: "kinetics_time_lp"

- A-optimality MILP solver time: "kinetics_time_milp"

- D-optimality NLP iterations and solver time: "kinetics_time_iter_nlp"

- D-optimality MINLP iterations and solver time: "kinetics_time_iter_minlp"


### Rotary bed case study 

At each budget, the FIM result and the optimal solution are stored separately in `pickle` files. 
Computational details including solver time and numbers of operations are also stored separately in `pickle` files.


#### FIM of optimal solutions

An example name: `LP_fim_1000_a`, the results of A-optimality LP problem of a budget of 1000 

Data file type: `pickle`, storing a numpy array of FIM of the shape Np*Np, Np is the number of parameters 

To replicate the results, iterate in the given budget range to retrieve the FIM stored in each data file

- A-optimality LP results: `rotary_results/LP_fim_x_a`, x in the range [1000, 2000, 3000, ..., 25000]

- A-optimality MILP results: `rotary_results/MILP_FIM_A_mip_x`, x in the range [1000, 2000, 3000, ..., 25000]

- D-optimality NLP results: `rotary_results/NLP_fim_x_d`, x in the range [1000, 2000, 3000, ..., 25000]

- D-optimality MINLP results: `rotary_results/MILP_fim_x_d_mip`, x in the range [1000, 2000, 3000, ..., 25000]

#### Optimal solutions

An example name: `LP_1000_a`, the results of A-optimality LP problem of a budget of 1000 

Data file type: `pickle`, storing a numpy array of the solutions of the shape Nm*Nm, Nm is the number of all measurements


- A-optimality LP results: `rotary_results/LP_x_a`, x in the range [1000, 2000, 3000, ..., 25000]

- A-optimality MILP results: `rotary_results/MILP_A_mip_x`, x in the range [1000, 2000, 3000, ..., 25000]

- D-optimality NLP results: `rotary_results/NLP_x_d`, x in the range [1000, 2000, 3000, ..., 25000]

- D-optimality MINLP results: `rotary_results/MILP_x_d_mip`, x in the range [1000, 2000, 3000, ..., 25000]

#### Computational details 

The computational details are stored separately. 

For A-optimality LP and MILP problems, the `pickle` files store a numpy array of the solver time of each budget 

For D-optimality NLP and MINLP problems, the `pickle` files store a dictionary, where the keys are the budgets. An example is: 

nlp_time={1000: {"t": 0.01, "n": 10}, ..., "5000": {"t": 0.01, "n": 10}}

For each budget, the value is a dictionary where the key `t` stores the solver time, `n` stores the number of iterations

- A-optimality LP solver time: `rotary_time_lp`

- A-optimality MILP solver time: `rotary_time_milp`

- D-optimality NLP iterations and solver time: `rotary_time_iter_nlp`

- D-optimality MINLP iterations and solver time: `rotary_time_iter_minlp`


