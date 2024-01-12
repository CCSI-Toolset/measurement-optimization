# measurement-opt

## Data files 

### Kinetics case study 

#### FIM of optimal solutions

- A-optimality LP results: `kinetics_results/May9_fim_x_a`, x in the range [1000, 1100, 1200, ..., 5000]

- A-optimality MILP results: `kinetics_results/May2_fim_x_a`, x in the range [1000, 1400, 1800, ..., 5000]

- D-optimality NLP results: `kinetics_results/May4_fim_x_d`, x in the range [1000, 1100, 1200, ..., 5000]

- D-optimality MINLP results: `kinetics_results/Dec9_fim_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

- Operating cost results: `kinetics_results/Dec12_fim_x_d_mip`, x in the range [1000, 1400, 1800, ..., 5000]

#### Optimal solutions

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

- A-optimality LP solver time: "rotary_time_lp"

- A-optimality MILP solver time: "rotary_time_milp"

- D-optimality NLP iterations and solver time: "rotary_time_iter_nlp"

- D-optimality MINLP iterations and solver time: "rotary_time_iter_minlp"

## Code 

- measure_optimize.py: Measurement optimization optimization framework

- greybox_generalize.py: Grey-box generalization 

- kinetics_MO.py: Kinetics case study

- rotary_bed_MO.py: Rotary bed case study

- Draw_figure.ipynb: Figure draw 

## Code running guidance 

## 
