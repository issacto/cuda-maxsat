#!/bin/bash -l
#SBATCH -n 30
#SBATCH --job-name=ga
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --gres=gpu
module load libs/cuda
lscpu
nvidia-smi
nvcc --version
nvcc  -o main main_64_reader.cu
echo "Randomly generated data"
for (( n=1; n<=10; n++ ))
do
    echo "$n round "
    ./main /users/k19029931/data/rand/rand64_$((n * 1000)).cnf 
done