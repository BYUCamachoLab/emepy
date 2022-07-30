#!/bin/bash

#SBATCH --time=48:00:00  # walltime
#SBATCH --ntasks=20   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=20G   # memory 
#SBATCH -J "emepy-meep"   # job name
#SBATCH --output=R-%x.%j.out

# Variables
export PATH="/fslhome/ihammond/miniconda3/bin:$PATH" # path to miniconda
source /fslhome/ihammond/miniconda3/etc/profile.d/conda.sh # source conda profile
conda activate pmeep3 # activate pmeep2 environment
srun python meep_only.py 0