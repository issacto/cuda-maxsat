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
count=0
for d in ./data/ubc-new/*; do
 echo $d
 timeout 1m ./main  $d 
 if [ $? -ne 1 ]; then 
    echo "the command timed out"
    let "count=count+1"
 fi
done
echo "Number of unsatisfied clauses:"
echo  $count