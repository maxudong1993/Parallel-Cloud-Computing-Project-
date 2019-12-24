#!/bin/bash
#SBATCH -p physical
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
module load Python/3.6.4-intel-2017.u2
export PYTHONIOENCODING=UTF_8
echo '2 nodes 8 cores'
time mpirun python3 Assignment_1.py
