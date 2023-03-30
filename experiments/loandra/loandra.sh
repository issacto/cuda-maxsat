#!/bin/bash -l
#SBATCH -n 30
#SBATCH --job-name=satSolver
#SBATCH --output=/scratch/users/%u/%j.out

lscpu

echo "Randomly generated data"
for (( n=1; n<=10; n++ ))
do
    echo "$n round - 1 minute"
    timeout 1m /users/k19029931/loandra/loandra /users/k19029931/data/rand/rand64_$((n * 1000)).cnf 
    echo "$n round - 5 minutes"
    timeout 5m /users/k19029931/loandra/loandra /users/k19029931/data/rand/rand64_$((n * 1000)).cnf 
done

echo "UBC benchmark data"
for (( n=1; n<=5; n++ ))
do
    echo "$n round - 1 minute"
    timeout 1m /users/k19029931/loandra/loandra /users/k19029931/data/ubc/uf50-0$((n * 100)).cnf 
    echo "$n round - 5 minutes"
    timeout 5m /users/k19029931/loandra/loandra /users/k19029931/data/ubc/uf50-0$((n * 100)).cnf
done
