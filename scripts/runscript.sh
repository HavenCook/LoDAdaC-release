#!/bin/bash -l
#SBATCH --job-name=airc
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4           # adjust if your data loader needs more cores
#SBATCH --partition=                # fill in
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:32g:4
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=               # fill in

# current parameters setup for 16 ranks, change nodes, ntasks-per-node, and gres accordingly

# load cuda, mpi, communication modules, might be different depending on cluster
# module load gcc/8.4.0/1
# module load cuda/10.2
# module load openmpi/4.0.3/1
# module load mxm
# module load hcoll
# module load knem

# run your environment initialization script here
# source $conda_setup
# conda activate LoDAdaC

srun --cpu_bind=cores python3 -u -m scripts.experiment