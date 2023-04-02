#include <stdio.h>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include <curand.h>
#include <curand_kernel.h>
#include <bits/stdc++.h>
#include <bitset>
#include <fstream>
#include <sstream>
#include <stdlib.h>
using namespace std;
using namespace std::chrono;
#define MAX 18446744073709551615
#define SATCLAUSES 3
/*
    Parameters
*/
#define threadsPerBlock 1024
#define N threadsPerBlock * 216 // perents
#define roundsPerMigration 10
#define threadsInBlockIsland 32                                                        // number of threads in an island
#define totalThreadsIsland N / threadsInBlockIsland                                    // number of islands
#define totalBlocksIsland (totalThreadsIsland + threadsPerBlock - 1) / threadsPerBlock // number of blocks for islands
#define selectionMode true                                                             // elitism or ranking selection
#define crossoverMode 2                                                                //  uniform or single-point or two-points crossover
#define mutationMode false                                                             // single or double mutation
#define mutationKeep false                                                             // exempt the best parent to be mutated
#define mutationThreshold 0.5                                                          // between 0 and 1
#define terminationMode true                                                           // terminate by rounds without improvement or by time
#define maxRound 20000
#define debugMode false // print rounds or print MAXSAT evaluation format

// https://stackoverflow.com/questions/65293876/cuda-gpuassert-an-illegal-memory-access-was-encountered
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__constant__ int d_satSize; // the number of clauses

__constant__ int d_maxBit; // variables (bits) needed for the problem set

__constant__ short d_satSets[10000 * SATCLAUSES]; // problem set literals

// initialise a state with a random state
__global__ void init(curandState_t *states, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        curand_init(clock64(), id, 0, &states[id]);
    }
}

// initialise a random integer number
__global__ void random_casting_int(curandState_t *states, int *numbers, int maxIndex, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        numbers[id] = curand_uniform(&states[id]) * maxIndex;
    }
}

// initialise a random float number
__global__ void random_casting_float(curandState_t *states, float *numbers, int maxIndex, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        numbers[id] = curand_uniform(&states[id]) * maxIndex;
    }
}

// initialise a parent which is of type unsigned long long int
__global__ void random_casting_parent(curandState_t *states, unsigned long long int *numbers, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        numbers[id] = curand_uniform_double(&states[id]) * MAX;
    }
}

// evaluate the fitness value of a chromosome
__global__ void evaluation(unsigned long long int *parents, unsigned int *parentVals, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        int tempVal = 0;
        for (int i = 0; i < d_satSize; i++)
        {
            for (int ii = 0; ii < SATCLAUSES; ii++)
            {
                if ((d_satSets[i * SATCLAUSES + ii] < 0 && (!((parents[id] >> abs(d_satSets[i * SATCLAUSES + ii]) - 1) & 1))) ||
                    (d_satSets[i * SATCLAUSES + ii] > 0 && ((parents[id] >> abs(d_satSets[i * SATCLAUSES + ii]) - 1) & 1)))
                {
                    // if sat sets literal is negative and chromosome index is 0 or sat sets literal is positive and chromosome index is 1
                    tempVal += 1;
                    break;
                }
            }
        }
        parentVals[id] = tempVal;
    }
}

__global__ void mutation(unsigned long long int *parents, float *mutateProb, int *mutateIndex, bool mode, bool isMutationKeep, int bestIndex, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        if (!isMutationKeep || id != bestIndex)
        {
            if (mode)
            {
                // Single Mutation
                if (mutateProb[id] > mutationThreshold)
                {
                    if (!((parents[id] >> mutateIndex[id]) & 1))
                    {
                        // if chromsome idth index is 0
                        parents[id] |= (1ULL << mutateIndex[id]);
                    }
                    else
                    {
                        // if chromsome idth index is 1
                        parents[id] &= ~(1ULL << mutateIndex[id]);
                    }
                }
            }
            else
            {
                // Double Mutation
                if (!((parents[id] >> mutateIndex[id]) & 1))
                {
                    // if chromsome idth index is 0
                    parents[id] |= (1ULL << mutateIndex[id]);
                }
                else
                {
                    // if chromsome idth index is 1
                    parents[id] &= ~(1ULL << mutateIndex[id]);
                }
                int nextId = id + 1;
                if (nextId > N)
                    nextId = nextId - N;
                if (mutateProb[id] > mutationThreshold)
                {
                    if (!((parents[id] >> mutateIndex[nextId]) & 1))
                    {
                        // if chromsome nextIdth index is 0
                        parents[id] |= (1ULL << mutateIndex[nextId]);
                    }
                    else
                    {
                        // if chromsome nextIdth index is 1
                        parents[id] &= ~(1ULL << mutateIndex[nextId]);
                    }
                }
            }
        }
    }
}

// crossover for single-point and two-points crossover
__global__ void crossover_fixed(unsigned long long int *parents, unsigned long long int *blockBestParents, int *splitIndex, int *length, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int startingPosition = splitIndex[id] - length[id];
    if (startingPosition < 0)
        startingPosition = 0;
    if (max > id)
    {
        int bId = blockIdx.x;
        for (int i = startingPosition; i < splitIndex[id]; i++)
        {
            if ((blockBestParents[bId] >> i) & 1)
            {
                // if selected chromsome ith index is 1
                if (!((parents[id] >> i) & 1))
                {
                    // if chromsome ith index is 0
                    parents[id] |= (1ULL << i);
                }
            }
            else
            {
                // if selected chromsome ith index is 0
                if ((parents[id] >> i) & 1)
                {
                    // if chromsome ith index is 1
                    parents[id] &= ~(1ULL << i);
                }
            }
        }
    }
}

__global__ void crossover_uniform(unsigned long long *parents, unsigned long long *blockBestParents, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        int bId = blockIdx.x;
        for (int i = 0; i < d_maxBit; i += 2)
        {
            if ((blockBestParents[bId] >> i) & 1)
            {
                // if selected chromsome ith index is 1
                if (!((parents[id] >> i) & 1))
                {
                    // if chromsome ith index is 0
                    parents[id] |= (1ULL << i);
                }
            }
            else
            {
                // if selected chromsome ith index is 0
                if ((parents[id] >> i) & 1)
                {
                    // if chromsome ith index is 1
                    parents[id] &= ~(1ULL << i);
                }
            }
        }
    }
}

__global__ void selection_elitism(unsigned long long int *parents, unsigned int *parentVals, unsigned long long int *blockBestParent, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        int bId = id * threadsInBlockIsland;
        unsigned int tmpLargestVal = 0;
        unsigned long long int tmpLargestPar = 0;
        // iterate over the threads in an island
        for (int i = 0; i < threadsInBlockIsland; i++)
        {
            if (parentVals[bId + i] > tmpLargestVal)
            {
                tmpLargestPar = parents[bId + i];
                tmpLargestVal = parentVals[bId + i];
            }
        }
        // select the chromosome with the highest fitness value at the corresponding blockBestParent array
        blockBestParent[id] = tmpLargestPar;
    }
}

__global__ void selection_wheel(unsigned long long int *parents, unsigned int *parentVals, unsigned long long int *blockBestParent, float *wheelProbs, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        int bId = id * threadsInBlockIsland;
        unsigned int tmpLowestVal = d_satSize + 100;
        unsigned int totalVal = 0;
        // find the lowest and total fitness value
        for (int i = 0; i < threadsInBlockIsland; i++)
        {
            if (parentVals[bId + i] < tmpLowestVal)
            {
                tmpLowestVal = parentVals[bId + i];
            }
            totalVal += parentVals[bId + i];
        }
        unsigned int base = totalVal - threadsInBlockIsland * tmpLowestVal;
        // store the cumulative proabability
        float tmpProb = 0;
        for (int i = 0; i < threadsInBlockIsland; i++)
        {
            tmpProb += (parentVals[bId + i] - tmpLowestVal) / base;
            if (tmpProb > wheelProbs[id])
            {
                // select the chromosome when the probability is higher than the randomly generated probability
                blockBestParent[id] = parents[bId + i];
                break;
            }
        }
    }
}

__global__ void internalReOrder(unsigned long long int *parents, unsigned int *parentVals, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        int bId = id * threadsInBlockIsland;
        int lowestIndex, highestIndex, highestVal = 0;
        int lowestVal = d_satSize + 100;
        ;
        for (int i = 0; i < threadsInBlockIsland; i++)
        {
            // store the chromsomes with the lowest and highest fitness values
            if (i == 0)
            {
                lowestVal = parentVals[bId + i];
                highestVal = parentVals[bId + i];
                lowestIndex = bId + i;
                highestIndex = bId + i;
            }
            else
            {
                if (parentVals[bId + i] < lowestVal)
                {
                    lowestVal = parentVals[bId + i];
                    lowestIndex = bId + i;
                }
                else if (parentVals[bId + i] > highestVal)
                {
                    highestVal = parentVals[bId + i];
                    highestIndex = bId + i;
                }
            }
        }
        unsigned long long int tmpLowest = parents[lowestIndex];
        unsigned long long int tmpHighest = parents[highestIndex];
        // swap the position of the first position with that of the chromosome with lowest fitness values
        parents[lowestIndex] = parents[bId];
        parents[bId] = tmpLowest;
        // swap the position of the last position with that of the chromosome with highest fitness values
        parents[highestIndex] = parents[bId + threadsInBlockIsland - 1];
        parents[bId + threadsInBlockIsland - 1] = tmpHighest;
    }
}

__global__ void migration(unsigned long long int *parents, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        // Migration - the last chromosome replaces the first chromosome of the next block
        int index = (id + 1) * threadsInBlockIsland - 1;
        if (index >= N)
            index = index - N;
        int replaceIndex = (id + 1) * threadsInBlockIsland;
        if (replaceIndex >= N)
            replaceIndex = replaceIndex - N;
        parents[replaceIndex] = parents[index];
    }
}

short *readSatSets(string fileName, int *h_maxBit, int *h_satSize)
{
    string tempText;
    // Read from the text file
    ifstream firstFileRead(fileName);
    ifstream secondFileRead(fileName);
    int tmpSatSize = 0;
    int tmpMaxBit = 0;
    while (getline(firstFileRead, tempText))
    {
        // process each line
        if (tempText[0] == 'p')
        {
            // if the line is about problem definition
            istringstream iss(tempText);
            string s;
            int tmpIndex = 0;
            while (getline(iss, s, ' '))
            {
                // process strings splitted by space in a line
                if (tmpIndex == 2)
                {
                    tmpMaxBit = atoi(s.c_str());
                }
                else if (tmpIndex == 3)
                {
                    tmpSatSize = atoi(s.c_str());
                }
                if (!(tmpIndex >= 2 && atoi(s.c_str()) == 0))
                    tmpIndex += 1;
            }
            break;
        }
    }
    bool isCount = false;
    int index = 0;
    short *h_satSets = new short[tmpSatSize * SATCLAUSES];
    while (getline(secondFileRead, tempText))
    {
        // process each line
        if (tempText[0] == 'p')
        {
            isCount = true;
        }
        else if (isCount && tempText[0] != 'c')
        {
            // iterate over all the literals in a line
            string tmpStr;
            for (int i = 0; i < tempText.size(); i++)
            {

                if (tempText[i] != ' ')
                {
                    tmpStr += tempText[i];
                }
                else
                {
                    if (tmpStr != "0" && !tmpStr.empty())
                    {
                        short tmpNumber = stoi(tmpStr);
                        h_satSets[index] = tmpNumber;
                        tmpStr = "";
                        index += 1;
                    }
                }
            }
        };
    }
    *h_maxBit = tmpMaxBit;
    *h_satSize = tmpSatSize;
    firstFileRead.close();
    secondFileRead.close();
    return h_satSets;
}

// print the bits of a chromosome
void printBits(unsigned long long int parent, int max)
{
    cout << parent << endl;
    for (int i = 0; i < max; i++)
    {
        if ((parent >> i) & 1)
        {
            cout << (i + 1);
        }
        else
        {
            cout << (i + 1) * -1;
        }
        cout << " ";
    }
    cout << endl;
}

void printOccupancyMetrics(int blockSize, int minGridSize, string funcName)
{
    cout << "--------------------------------------" << endl;
    cout << "Function Name: " << funcName << endl;
    cout << "Block Size: " << blockSize << endl;
    cout << "Min Grid Size: " << minGridSize << endl;
}

// print the max block and grid size for all the kernel functions in the program
void printMaxBlockSize()
{
    int blockSize;
    int minGridSize;
    int max_active_blocks;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    cout << "Number of Streaming Multiprocessor: " << num_sm << endl;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, init, blockSize, 0);
    cout << "Maximum active blocks:  " << max_active_blocks << endl;
    cout << "--------------------------------------" << endl;
    printOccupancyMetrics(blockSize, minGridSize, "init");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, random_casting_int, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "random_casting_int");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, random_casting_float, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "random_casting_float");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, random_casting_parent, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "random_casting_parent");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, evaluation, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "evaluation");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mutation, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "mutation");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, crossover_fixed, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "crossover_fixed");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, crossover_uniform, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "crossover_uniform");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, selection_elitism, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "selection_elitism");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, selection_wheel, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "selection_wheel");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, internalReOrder, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "internalReOrder");
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, migration, 0, 0);
    printOccupancyMetrics(blockSize, minGridSize, "migration");
    cout << "--------------------------------------" << endl;
}

int main(int argc, char **argv)
{
    // printMaxBlockSize();
    srand(time(0));
    auto start = high_resolution_clock::now();
    string fileName = "";
    if (argc == 2)
        fileName = argv[1];
    else if (argc > 2)
        std::invalid_argument("too many arguments");
    else
        throw std::invalid_argument("no file specified");
    if (!debugMode)
    {
        cout << "c ------------------------------------" << endl;
        cout << "c CUDA Genetic Algorithm MAXSAT solver" << endl;
        cout << "c ------------------------------------" << endl;
    }

    int h_maxBit = 0;
    int h_satSize = 0;
    short *h_satSets = readSatSets(fileName, &h_maxBit, &h_satSize);
    cudaMemcpyToSymbol(d_maxBit, &h_maxBit, 1 * sizeof(int));
    cudaMemcpyToSymbol(d_satSize, &h_satSize, 1 * sizeof(int));
    cudaMemcpyToSymbol(d_satSets, h_satSets, h_satSize * SATCLAUSES * sizeof(short));

    curandState_t *d_parent_states;
    gpuErrchk(cudaMalloc((void **)&d_parent_states, N * sizeof(curandState_t)));
    unsigned long long int *d_parents;
    gpuErrchk(cudaMalloc(&d_parents, N * sizeof(unsigned long long int)));

    // initialise states for parents
    init<<<N / threadsPerBlock, threadsPerBlock>>>(d_parent_states, N);
    gpuErrchk(cudaPeekAtLastError());

    // initialise values for parents
    random_casting_parent<<<N / threadsPerBlock, threadsPerBlock>>>(d_parent_states, d_parents, N);
    gpuErrchk(cudaPeekAtLastError());

    // initialise an array to store fitness values of parents in device
    unsigned int *d_parentVals;
    gpuErrchk(cudaMalloc(&d_parentVals, N * sizeof(unsigned int)));

     // initialise arrays to store fitness values of parents in host
    unsigned long long int *h_parents;
    h_parents = (unsigned long long int *)malloc(N * sizeof(unsigned long long int));

    // initialise an array to store fitness values of parents in device
    unsigned int *h_parentVals;
    h_parentVals = (unsigned int *)malloc(N * sizeof(unsigned int));
    unsigned long long int *d_block_bests;
    gpuErrchk(cudaMalloc(&d_block_bests, totalBlocksIsland * sizeof(unsigned long long int)));

    // initialise all the device arrays
    curandState_t *d_crossover_states, *d_crossover_length_states, *d_mutation_index_states, *d_mutation_prob_status, *d_selection_prob_status;
    int *d_crossover_index, *d_crossover_length_index, *d_mutation_index;
    float *d_selection_prob, *d_mutation_prob;
    // crossover index
    gpuErrchk(cudaMalloc((void **)&d_crossover_states, N * sizeof(curandState_t)));
    gpuErrchk(cudaMalloc(&d_crossover_index, N * sizeof(int)));
    // crossover length
    gpuErrchk(cudaMalloc((void **)&d_crossover_length_states, N * sizeof(curandState_t)));
    gpuErrchk(cudaMalloc(&d_crossover_length_index, N * sizeof(int)));
    // mutation index
    gpuErrchk(cudaMalloc((void **)&d_mutation_index_states, N * sizeof(curandState_t)));
    gpuErrchk(cudaMalloc(&d_mutation_index, N * sizeof(int)));
    // mutation probability
    gpuErrchk(cudaMalloc((void **)&d_mutation_prob_status, N * sizeof(curandState_t)));
    gpuErrchk(cudaMalloc(&d_mutation_prob, N * sizeof(float)));
    // selection probability
    gpuErrchk(cudaMalloc((void **)&d_selection_prob_status, totalThreadsIsland * sizeof(curandState_t)));
    gpuErrchk(cudaMalloc(&d_selection_prob, N * sizeof(float)));

    // variables
    unsigned int res_maxParentVal = 0;
    unsigned long long int res_maxParent = 0;
    auto res_maxTime = start;
    int roundsWithoutImprovement = 0;
    int res_maxRound = 0;
    int roundIndex = 0;

    // while loop starts here
    while ((terminationMode && (roundsWithoutImprovement < maxRound)))
    {
        if (roundIndex % roundsPerMigration == (roundsPerMigration / 2))
        {
            // Migration
            if (debugMode)
                cout << "<- Migration ->" << endl;
            internalReOrder<<<totalBlocksIsland, threadsPerBlock>>>(d_parents, d_parentVals, totalThreadsIsland);
            gpuErrchk(cudaPeekAtLastError());
            evaluation<<<N / threadsPerBlock, threadsPerBlock>>>(d_parents, d_parentVals, N);
            gpuErrchk(cudaPeekAtLastError());
            /* migration */
            migration<<<totalBlocksIsland, threadsPerBlock>>>(d_parents, totalThreadsIsland);
            gpuErrchk(cudaPeekAtLastError());
        }
        evaluation<<<N / threadsPerBlock, threadsPerBlock>>>(d_parents, d_parentVals, N);
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(h_parentVals, d_parentVals, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_parents, d_parents, N * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

        // selection
        if (selectionMode)
        {
            // Elitism selection
            selection_elitism<<<totalBlocksIsland, threadsPerBlock>>>(d_parents, d_parentVals, d_block_bests, totalThreadsIsland);
            gpuErrchk(cudaPeekAtLastError());
        }
        else
        {
            // Roulette Wheel Selection
            init<<<totalBlocksIsland, threadsPerBlock>>>(d_selection_prob_status, totalThreadsIsland);
            gpuErrchk(cudaPeekAtLastError());
            random_casting_float<<<totalBlocksIsland, threadsPerBlock>>>(d_selection_prob_status, d_selection_prob, 1, totalThreadsIsland);
            gpuErrchk(cudaPeekAtLastError());
            selection_wheel<<<totalBlocksIsland, threadsPerBlock>>>(d_parents, d_parentVals, d_block_bests, d_selection_prob, totalThreadsIsland);
            gpuErrchk(cudaPeekAtLastError());
        }
        gpuErrchk(cudaPeekAtLastError());

        // Find the highest fitness value in that round
        int tempLargestParentIndex = 0;
        unsigned int tempLargestParentValue = 0;
        evaluation<<<N / threadsPerBlock, threadsPerBlock>>>(d_parents, d_parentVals, N);
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(h_parentVals, d_parentVals, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_parents, d_parents, N * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++)
        {
            // If the chromsome with the highest fitness value in that round is found, store the value
            if (h_parentVals[i] > tempLargestParentValue)
            {
                tempLargestParentValue = h_parentVals[i];
                tempLargestParentIndex = i;
            }
        }

        if (debugMode)
        {
            // If it is in debug mode, print the highest fitness value each round
            cout << roundIndex << ". "
                 << "answer: ";
            cout << tempLargestParentValue << "  parent: ";
            cout << h_parents[tempLargestParentIndex] << " ";
            cout << "at " << tempLargestParentIndex << endl;
        }

        if (tempLargestParentValue > res_maxParentVal)
        {
            // If the chromsome with the highest fitness value so far is found, end the program
            res_maxParentVal = tempLargestParentValue;
            res_maxParent = h_parents[tempLargestParentIndex];
            res_maxRound = roundIndex;
            res_maxTime = high_resolution_clock::now();
            roundsWithoutImprovement = -1;
            if (!debugMode)
            {
                if (h_satSize == res_maxParentVal)
                {
                    // If all the clauses are satisfied
                    cout << "s OPTIMUM FOUND" << endl;
                    cout << "v ";
                    printBits(res_maxParent, h_maxBit);
                    cout << endl;
                    break;
                }
                else
                {
                    cout << "o " << (h_satSize - res_maxParentVal) << endl;
                }
            }
        }

        // crossover
        if (crossoverMode)
        {
            // single-point crossover
            init<<<N / threadsPerBlock, threadsPerBlock>>>(d_crossover_states, N);
            gpuErrchk(cudaPeekAtLastError());
            random_casting_int<<<N / threadsPerBlock, threadsPerBlock>>>(d_crossover_states, d_crossover_index, h_maxBit, N);
            gpuErrchk(cudaPeekAtLastError());
            crossover_fixed<<<N / threadsPerBlock, threadsPerBlock>>>(d_parents, d_block_bests, d_crossover_index, d_crossover_index, N);
        }
        else if (crossoverMode == 2)
        {
            // two-point crossover
            init<<<N / threadsPerBlock, threadsPerBlock>>>(d_crossover_states, N);
            gpuErrchk(cudaPeekAtLastError());
            random_casting_int<<<N / threadsPerBlock, threadsPerBlock>>>(d_crossover_states, d_crossover_index, h_maxBit, N);
            gpuErrchk(cudaPeekAtLastError());
            init<<<N / threadsPerBlock, threadsPerBlock>>>(d_crossover_length_states, N);
            gpuErrchk(cudaPeekAtLastError());
            random_casting_int<<<N / threadsPerBlock, threadsPerBlock>>>(d_crossover_length_states, d_crossover_length_index, h_maxBit, N);
            gpuErrchk(cudaPeekAtLastError());
            crossover_fixed<<<N / threadsPerBlock, threadsPerBlock>>>(d_parents, d_block_bests, d_crossover_index, d_crossover_length_index, N);
        }
        else
        {
            // uniform crossover
            crossover_uniform<<<N / threadsPerBlock, threadsPerBlock>>>(d_parents, d_block_bests, N);
        }
        gpuErrchk(cudaPeekAtLastError());

        // mutation
        // initialise new random states and numbers for mutation
        init<<<N / threadsPerBlock, threadsPerBlock>>>(d_mutation_index_states, N);
        gpuErrchk(cudaPeekAtLastError());
        random_casting_int<<<N / threadsPerBlock, threadsPerBlock>>>(d_mutation_index_states, d_mutation_index, h_maxBit, N);
        gpuErrchk(cudaPeekAtLastError());
        init<<<N / threadsPerBlock, threadsPerBlock>>>(d_mutation_prob_status, N);
        gpuErrchk(cudaPeekAtLastError());
        random_casting_float<<<N / threadsPerBlock, threadsPerBlock>>>(d_mutation_prob_status, d_mutation_prob, 1, N);
        gpuErrchk(cudaPeekAtLastError());
        mutation<<<N / threadsPerBlock, threadsPerBlock>>>(d_parents, d_mutation_prob, d_mutation_index, mutationMode, mutationKeep, tempLargestParentIndex, N);
        gpuErrchk(cudaPeekAtLastError());

        roundIndex += 1;
        roundsWithoutImprovement += 1;
    }
    /*
        Print Final Result
    */
    if (debugMode)
    {
        cout << "-------------LARGEST PARENT-------------" << endl;
        cout << "Number of bits : " << h_maxBit << endl;
        cout << "SAT Size: " << h_satSize << endl;
        cout << "Best parent: " << res_maxParent << endl;
        cout << "Best parent: " << bitset<64>(res_maxParent).to_string() << endl;
        cout << "Best value: " << res_maxParentVal << endl;
        cout << "Best round: " << res_maxRound << endl;
        auto duration_max = duration_cast<microseconds>(res_maxTime - start);
        cout << "Best round time: " << duration_max.count() << endl;
        cout << "Total round: " << roundIndex << endl;
        cout << "----------------------------------" << endl;
        cout << "Selection Mode: ";
        if (selectionMode)
            cout << "Elitism" << endl;
        else
            cout << "Roulette Wheel" << endl;
        cout << "Crossover Mode: ";
        if (crossoverMode==1)
            cout << "Single Point" << endl;
        else if(crossoverMode==2)
            cout << "Two-Points" << endl;
        else
            cout << "Uniform" << endl;
        cout << "Mutation Mode: ";
        if (mutationMode)
            cout << "Single" << endl;
        else
            cout << "Double" << endl;
        cout << "Termination Mode: ";
        if (terminationMode)
            cout << "Rounds" << endl;
        else
            cout << "Time" << endl;
    }

    /* Free device memory */
    cudaFree(d_parents);
    cudaFree(d_block_bests);
    cudaFree(d_parentVals);
    cudaFree(d_parent_states);
    cudaFree(d_selection_prob_status);
    cudaFree(d_crossover_states);
    cudaFree(d_crossover_length_states);
    cudaFree(d_mutation_index_states);
    cudaFree(d_mutation_prob_status);
    cudaFree(d_selection_prob);
    cudaFree(d_crossover_index);
    cudaFree(d_crossover_length_index);
    cudaFree(d_mutation_index);
    cudaFree(d_mutation_prob);

    /* Free host memory */
    free(h_parents);
    free(h_parentVals);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: "
         << duration.count() << " microseconds" << endl;
    if (h_satSize == res_maxParentVal)
        return 1;
    else
        return 0;
}