#!/bin/bash
#SBATCH -p physical
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
module load Python/3.6.4-intel-2017.u2
export PYTHONIOENCODING=UTF_8
echo '1 node 1 core'
time mpirun python3 Assignment_1.py
