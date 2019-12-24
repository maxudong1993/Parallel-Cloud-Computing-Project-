#!/bin/bash
#SBATCH -p physical
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
module load Python/3.6.4-intel-2017.u2
export PYTHONIOENCODING=UTF_8
echo '1 node 8 cores'
time mpirun python3 Assignment_1.py
