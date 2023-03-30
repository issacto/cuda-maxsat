#!/bin/bash -l
#SBATCH -n 30
#SBATCH --job-name=satSolver
#SBATCH --output=/scratch/users/%u/%j.out
lscpu
cd ..
echo "Randomly generated data"
for (( n=1; n<=10; n++ ))
do
    echo "$n round - 1 hour"
    timeout 1h /users/k19029931/open-wbo/open-wbo /users/k19029931/data/rand/rand64_$((n * 1000)).cnf 
done
