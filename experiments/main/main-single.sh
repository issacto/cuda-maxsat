#!/bin/bash -l
#SBATCH -n 30
#SBATCH --job-name=ga
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --gres=gpu
module load libs/cuda
lscpu
nvidia-smi
nvcc --version
nvcc -o main main_64_reader.cu
timeout 5m ./main ./cuda10000.cnf