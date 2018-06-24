#include <cstdlib>
#include <vector>
#include <chrono>
#include <iostream>>
#include <cmath>
#include <numeric>
#include <math.h>

// GPU Libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Macro to handle errors occured in CUDA api
#define funcCheck(stmt) do { cudaError_t err = stmt; if (err != cudaSuccess) { printf("[ERROR] Failed to run stmt %d, error body: %s\n", __LINE__, cudaGetErrorString(err)); return -1; } } while (0)

#define N 4194304

// Define some execution parameters
#define BLOCK_SIZE 1024

using namespace std;

int parallelReduction(int *a);
void printArray(int *a, int n);


__device__ void
recursiveReduce(int *g_inData, int *g_outData, int inSize, int outSize)
{
	extern __shared__ int sData[];

	// Identification
	unsigned int tId = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialize
	sData[tId] = 0;

	__syncthreads();

	// Fill up the shared memory
	if (tId < blockDim.x) {
		sData[tId] = g_inData[i];
	}

	__syncthreads();

	// Tree based reduction
	for (unsigned int d = 1; d < blockDim.x; d *= 2) {
		if (tId % (2 * d) == 0)
			if (tId + d < blockDim.x)
				sData[tId] += sData[tId + d];

		__syncthreads();
	}

	// Write the result for this block to global memory
	if (tId == 0) {
		g_outData[blockIdx.x] = sData[0];
	}


	__syncthreads();

	// Recursive call
	if (outSize > 1 && i == 0) {

		// Kernel Launch
		recursiveReduce(g_outData, g_outData, outSize, (outSize - 1) / blockDim.x + 1);

	}
	else return;

}

__global__ void
reduceKernel(int *g_inData, int *g_outData, int inSize, int outSize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i == 0) {
		recursiveReduce(g_inData, g_outData, inSize, outSize);
	}
}

int main(int argc, char *argv[]) {
	// initialize a vector of size N with 1
	vector<int> v(N, 1);

	// parallelReduction call
	parallelReduction(&v[0]);

	// capture start time
	auto start_time = chrono::high_resolution_clock::now();

	// reduction
	auto sum = accumulate(begin(v), end(v), 0);

	// capture end time
	auto end_time = chrono::high_resolution_clock::now();

	// elapsed time in milliseconds
	auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

	// print sum and elapsed time
	cout << "[-] Serial Sum: " << sum << endl;
	cout << "[-] Serial Duration: " << duration.count() << "ms" << endl;

	system("pause");
	return EXIT_SUCCESS;
}

int parallelReduction(int *a)
{
	// Define the allocation pointers
	int *d_A;
	int *d_Blocks;

	// Define the proper size
	int memSize = N * sizeof(int);

	// Allocate space on GPU
	funcCheck(cudaMalloc((void **)&d_A, memSize));

	// Copy data to GPU
	funcCheck(cudaMemcpy(d_A, a, memSize, cudaMemcpyHostToDevice));

	// Calculate execution parameters
	int grid_x = ceil((float) N / BLOCK_SIZE);
	int block_x = BLOCK_SIZE;

	// Define the proper size for blocks array
	int blocksMemSize = grid_x * sizeof(int);

	// Allocate space on GPU (blocks)
	funcCheck(cudaMalloc((void **)&d_Blocks, blocksMemSize));

	// Setup execution parameters
	dim3 gridDimensions(grid_x, 1, 1);
	dim3 blockDimensions(block_x, 1, 1);

	printf("Computing result using CUDA Kernel...\n");
	// Setup the time
	cudaEvent_t start;
	funcCheck(cudaEventCreate(&start));
	cudaEvent_t stop;
	funcCheck(cudaEventCreate(&stop));

	// Start the timer
	funcCheck(cudaEventRecord(start, NULL));

	// KERNEL LAUNCH
	reduceKernel << <gridDimensions, blockDimensions, block_x * sizeof(int) >> > (d_A, d_Blocks, memSize, grid_x);

	// Kernel launch error handling
	funcCheck(cudaGetLastError());

	// Stop the timer
	funcCheck(cudaEventRecord(stop, NULL));

	// Wait for the stop event to complete
	funcCheck(cudaEventSynchronize(stop));

	// Calculate the elapsed time
	float elapsed_time = 0.0f;
	funcCheck(cudaEventElapsedTime(&elapsed_time, start, stop));
	printf("Elapsed time in msec = %f\n", elapsed_time);

	// Copy result back to host
	int *h_Blocks = (int *)malloc(blocksMemSize);
	funcCheck(cudaMemcpy(h_Blocks, d_Blocks, blocksMemSize, cudaMemcpyDeviceToHost));
	//printf("Parallel computation result: \n");
	//printArray(h_Blocks, grid_x);

	// Free the allocated spaces
	cudaFree(d_A);
	cudaFree(d_Blocks);

	return EXIT_SUCCESS;
}

void printArray(int *a, int n)
{
	for (int i = 0; i < n; i++) {
		printf("%d  ", a[i]);
	}
	printf("\n");
}
