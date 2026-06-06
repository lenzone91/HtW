#!/bin/bash
#SBATCH --job-name=XXX
#SBATCH --output=XXX.out.%j
#SBATCH --error=XXX.err.%j
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --partition=XXX
#SBATCH -t 71:59:00
#SBATCH --wckey=XXX

export OMP_NUM_THREADS=1
export OMP_PLACES=cores

srun python XXX.py


