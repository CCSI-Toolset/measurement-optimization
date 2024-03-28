#!/bin/bash
#$ -q long           	    # Specify queue
#$ -pe smp 24               # Specify number of cores to use.
#$ -N rotary_bed_tests1     # Specify job name
#$ -M adowling@nd.edu	    # Specify email
#$ -m bae		            # Send email beginning, abort, end

conda activate measurement_optimization

#module load python
module load gurobi/10.0.2
module load ipopt/hsl/3.12.8

python3 rotary_bed_MO.py
