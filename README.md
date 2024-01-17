# Measurement Optimization 

This repository contains code and results for paper: Measure this, not that: Optimizing the cost and model-based information content of measurements 

## Installation instructions 

How to create an environment to run code: 

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
- a generalization version is under working. For now, we use two different branches for each case study:
- branch for kinetic case study: 

  `pip install git+https://github.com/jialuw96/pyomo.git@MindtpyReactor`

- branch for rotary bed case study: 

  `pip install git+https://github.com/jialuw96/pyomo.git@MindtpyRotary`
   
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


## Example to run code for case studies

Setup the scripts to reproduce results from the paper: 

### Kinetics case study 

- Step 1: run `kinetics_MO.py`
- Step 2: change budgets with the variable `budget_opt`
- Step 3: use draw_figure.ipynb

### Rotary bed case study 

run `rotary_bed_MO.py`
- change budgets with the variable `budget_opt`

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

#### FIM of final results 

An example name: `May9_fim_1000_a`, the results of A-optimality LP problem of a budget of 1000 

Data file type: `pickle`, storing a numpy array of FIM

- A-optimality LP results: `kinetics_results/May9_fim_x_a`, x in the range [1000, 1100, 1200, ..., 5000]

- A-optimality MILP results: `kinetics_results/May2_fim_x_a`, x in the range [1000, 1400, 1800, ..., 5000]

- D-optimality NLP results: `kinetics_results/May4_fim_x_d`, x in the range [1000, 1100, 1200, ..., 5000]

- D-optimality MINLP results: `kinetics_results/Dec9_fim_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

- Operating cost results: `kinetics_results/Dec12_fim_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

#### Optimal solutions

An example name: `May9_1000_a`, the results of A-optimality LP problem of a budget of 1000 

Data file type: `pickle`, storing a numpy array of the solutions

- A-optimality LP results: `kinetics_results/May9_x_a`, x in the range [1000, 1100, 1200, ..., 5000]

- A-optimality MILP results: `kinetics_results/May2_x_a`, x in the range [1000, 1400, 1800, ..., 5000]

- D-optimality NLP results: `kinetics_results/May4_x_d`, x in the range [1000, 1100, 1200, ..., 5000]

- D-optimality MINLP results: `kinetics_results/Dec9_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

- Operating cost results: `kinetics_results/Dec12_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

#### Computational details 


- A-optimality LP solver time: "kinetics_time_lp"

- A-optimality MILP solver time: "kinetics_time_milp"

- D-optimality NLP iterations and solver time: "kinetics_time_iter_nlp"

- D-optimality MINLP iterations and solver time: "kinetics_time_iter_minlp"


### Rotary bed case study 

#### FIM of optimal solutions

- A-optimality LP results: `rotary_results/May12_fim_x_a`, x in the range [1000, 2000, 3000, ..., 25000]

- A-optimality MILP results: `rotary_results/Apr17_FIM_A_mip_x`, x in the range [1000, 2000, 3000, ..., 25000]

- D-optimality NLP results: `rotary_results/May10_fim_x_d`, x in the range [1000, 2000, 3000, ..., 25000]

- D-optimality MINLP results: `rotary_results/Dec7_fim_x_d_mip`, x in the range [1000, 2000, 3000, ..., 25000]

#### Optimal solutions

- A-optimality LP results: `rotary_results/May12_x_a`, x in the range [1000, 2000, 3000, ..., 25000]

- A-optimality MILP results: `rotary_results/Apr17_A_mip_x`, x in the range [1000, 2000, 3000, ..., 25000]

- D-optimality NLP results: `rotary_results/May10_x_d`, x in the range [1000, 2000, 3000, ..., 25000]

- D-optimality MINLP results: `rotary_results/Dec7_x_d_mip`, x in the range [1000, 2000, 3000, ..., 25000]

#### Computational details 

- A-optimality LP solver time: `rotary_time_lp`

- A-optimality MILP solver time: `rotary_time_milp`

- D-optimality NLP iterations and solver time: `rotary_time_iter_nlp`

- D-optimality MINLP iterations and solver time: `rotary_time_iter_minlp`


