# CUDA-MAXSAT

An island-based genetic algorithm using Nvidia’s CUDA is purposed to solve 3-MAXSAT problems.

## Data
Download 3-SAT benchmark data at https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html
<br /> Alternatively, generate a MAXSAT problem with the generator.
```
g++ -o generator generator.cpp
 ./generator <clause size> <number of variables> <number of clauses>
```

## Solver
The code for the genetic algorithm is included in the main.cu file.
```
nvcc  -o main main.cu
./main <file name>
```
## Verifier
Paste the selected chromosome inside the file and verify the result with the file name specified in the command line.
```
g++ -o verfier verfier.cpp
./verfier <file name>
```
## KCL Script

There are a few bash scripts situated in the experiments directory to run experiments on the KCL HPC cluster.