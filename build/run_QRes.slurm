#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -c 4
#SBATCH --mem=8GB
#SBATCH --account=education-eemcs-courses-wi4450 
#SBATCH --reservation=wi4450

n=23
l=1
export OMP_NUM_THREADS=4
srun ./QRes_3b1d $n $n $l $l
