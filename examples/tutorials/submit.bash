#!/bin/bash

#SBATCH --time=2:00:00  # walltime
#SBATCH --ntasks=20   # number of processor cores (i.e. tasks)
#SBATCH --nodes=5   # number of nodes
#SBATCH --mem=120G   # memory 
#SBATCH -J "eme"   # job name
#SBATCH --output=R-%x.%j.out



# Variables
export PATH="/fslhome/ihammond/miniconda3/bin:$PATH"
source /fslhome/ihammond/miniconda3/etc/profile.d/conda.sh
conda activate emepy
module load mpi/openmpi-1.10.7_gcc-9.2.0
mpirun -np 20 python3 -m mpi4py adiabatic.py
# python3 optimizer.py
