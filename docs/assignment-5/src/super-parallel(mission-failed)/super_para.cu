
// Let it be.
#define _CRT_SECURE_NO_WARNINGS
#define BLOCK_SIZE 4
#define NUM_OF_ADDS_PER_THREAD 2
#define WARP_SIZE 4
#define GRID_SIZE 4

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// CUDA Runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void fill_array(int *a, size_t n);
void prefix_sum_serial(int *a, size_t n);
void print_array(int *a, size_t n);
float compute_mse(int *a, int *b, int n);
int compute_prefix_sum(int *a, size_t n);

__global__ void
computeSubArraysKernel(int *a, size_t n)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;		// Global thread ID

	int threadScopeStart = globalId * NUM_OF_ADDS_PER_THREAD;		// Job start for each thread
	int threadScopeEnd = threadScopeStart + NUM_OF_ADDS_PER_THREAD;		// Job end for each thread

	/* Each thread scans its own sub array */
	for (int i = threadScopeStart + 1; i < threadScopeEnd; i++) {
		if (i >= n)
			break;

		a[i] += a[i - 1];
	}
}

__global__ void
computeBlocksKernel(int *a, size_t mainArraySize, int *subArraysBiggestValuesArray, size_t subArraysBiggestValuesArraySize, int *helperArray)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;		// Global thread ID

	int threadScopeStart = globalId * NUM_OF_ADDS_PER_THREAD;		// Job start for each thread
	int threadScopeEnd = threadScopeStart + NUM_OF_ADDS_PER_THREAD;		// Job end for each thread


	// Each thread copies the final value of its sub array to an intermediate array (Happens inside each block)
	if (globalId < subArraysBiggestValuesArraySize) {		// Valid write on subArraysBiggestValuesArray
		subArraysBiggestValuesArray[threadIdx.x] = a[threadScopeEnd - 1];
	}

	// Wait until all subarrays are copied into shared memory
	__syncthreads();

	// Scan on subArraysBiggestValuesArray
	// Done by one thread in each block
	if (threadIdx.x == 0) {
		for (int i = 1; i < BLOCK_SIZE; i++) {
			subArraysBiggestValuesArray[i] += subArraysBiggestValuesArray[i - 1];
		}
	}

	// printf("subArraysBiggestValuesArray[%d]: %d\n", globalId, subArraysBiggestValuesArray[threadIdx.x]);
	__syncthreads();

	// printf("blocksBiggestValuesArray[%d]: %d\n", blockIdx.x, blocksBiggestValuesArray[blockIdx.x]);
	// Update block size
	int UPDATED_BLOCK_SIZE = (BLOCK_SIZE - 1) / NUM_OF_ADDS_PER_THREAD + 1;
	if (threadIdx.x == UPDATED_BLOCK_SIZE - 1) {
		helperArray[blockIdx.x] = subArraysBiggestValuesArray[threadIdx.x];
		
		// Make results visible to everyone :D
		
		//__syncthreads();
	}
	__syncthreads();

	if (globalId == 0) {
		for (int k = 1; k < GRID_SIZE; k++) {
			helperArray[k] = helperArray[k] + helperArray[k - 1];
		}
	}
}

__global__ void
reduceResultsKernel(int *a, size_t mainArraySize, int *subArraysBiggestValuesArray, size_t subArraysBiggestValuesArraySize, int *helperArray)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;		// Global thread ID

	int threadScopeStart = globalId * NUM_OF_ADDS_PER_THREAD;		// Job start for each thread
	int threadScopeEnd = threadScopeStart + NUM_OF_ADDS_PER_THREAD;		// Job end for each thread

	//printf("subArraysBiggestValuesArray[%d]: %d\n, block[%d]\n", threadIdx.x, helperArray[blockIdx.x], blockIdx.x)
}


void scanCPU(float *f_out, float *f_in, int i_n)
{
	f_out[0] = 0;
	for (int i = 1; i < i_n; i++)
		f_out[i] = f_out[i - 1] + f_in[i - 1];
}

int main(int argc, char *argv[]) {
	// Input N
	size_t n = 0;
	printf("[-] Please enter N: ");
	scanf("%uld\n", &n);
	// Allocate memory for array
	int * a = (int *)malloc(n * sizeof a);
	// Fill array with numbers 1..n
	fill_array(a, n);
	// Print array
	// print_array(a, n);
	// Compute prefix sum
	// prefix_sum(a, n);

	// Create a copy of a for comparison
	int * b = (int *)malloc(n * sizeof b);
	for (int i = 0; i < n; i++) {
		b[i] = a[i];
	}

	 print_array(b, n);

	// Compute the serial prefix sum on b
	prefix_sum_serial(b, n);

	// Compute the parallel prefix sum on a
	compute_prefix_sum(a, n);

	// Find the computation error 
	float error = compute_mse(a, b, n);

	printf("Computation error is %lf\n", error);

	// Print array
	 print_array(a, n);

	// Free allocated memory
	free(a);

	system("pause");
	return EXIT_SUCCESS;
}


int compute_prefix_sum(int *a, size_t n) {

	// To be allocated on device memory
	int* d_A;
	int* d_subArrays;
	int* d_HelperArray;

	// Handles errors
	cudaError_t error;

	if (a == NULL)
	{
		fprintf(stderr, "Failed to allocate host vector a!\n");
		exit(EXIT_FAILURE);
	}

	// Allocate memory on device for d_A
	int memSize = n * sizeof(int);

	error = cudaMalloc((void **)&d_A, memSize);

	if (error != cudaSuccess) {
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Allocate memory on device d_subArraysBiggestValuesArray;
	int subArraysBiggestValuesArraySize = ceil((float)n / NUM_OF_ADDS_PER_THREAD);

	// Allocate memory on device d_subArraysBiggestValuesArray;
	error = cudaMalloc((void **)&d_subArrays, subArraysBiggestValuesArraySize);

	if (error != cudaSuccess) {
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}


	// Allocate memory on device d_subArraysBiggestValuesArray;
	error = cudaMalloc((void **)&d_HelperArray, GRID_SIZE);

	if (error != cudaSuccess) {
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}



	// Copy data from host to device
	error = cudaMemcpy(d_A, a, n * sizeof(int), cudaMemcpyHostToDevice);

	if (error != cudaSuccess) {
		printf("cudaMemcpy d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	printf("Processing data on GPU...\n");

	// Setup CUDA events for timings
	cudaEvent_t start;

	// Create the CUDA events
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}


	// Allocate memory for the subArraySums array
	int NUM_BLOCKS = ceil((float)n / BLOCK_SIZE);
	int NUM_THREADS_IN_BLOCK = BLOCK_SIZE / NUM_OF_ADDS_PER_THREAD;


	dim3 gridDimensionsMapKernel(GRID_SIZE, 1, 1);
	dim3 blockDimensionsMapKernel(NUM_THREADS_IN_BLOCK, 1, 1);

	computeSubArraysKernel << <gridDimensionsMapKernel, blockDimensionsMapKernel >> > (d_A, n);
	computeBlocksKernel << <gridDimensionsMapKernel, blockDimensionsMapKernel, subArraysBiggestValuesArraySize >> > (d_A, n, d_subArrays, subArraysBiggestValuesArraySize, d_HelperArray);
	reduceResultsKernel << <gridDimensionsMapKernel, blockDimensionsMapKernel >> > (d_A, n, d_subArrays, subArraysBiggestValuesArraySize, d_HelperArray);
	
	// Check kernel launch
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}


	// Wait for the stop event to be completed by all threads
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Compute the elapsed time
	float elapsed_time = 0.0f;
	error = cudaEventElapsedTime(&elapsed_time, start, stop);

	printf("Elapsed time in msec = %lf\n", elapsed_time);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	error = cudaMemcpy(a, d_A, n * sizeof(int), cudaMemcpyDeviceToHost);


	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_A);

	return EXIT_SUCCESS;
}
void prefix_sum_serial(int *a, size_t n) {
	int i;
	for (i = 1; i < n; ++i) {
		a[i] = a[i] + a[i - 1];
	}
}

void print_array(int *a, size_t n) {
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i) {
		printf("%d, ", a[i]);
	}
	printf("\b\b \n");
}

void fill_array(int *a, size_t n) {
	int i;
	for (i = 0; i < n; ++i) {
		a[i] = i + 1;
	}
}

float compute_mse(int *a, int *b, int n) {
	float err;

	for (int i = 0; i < n; i++) {
		err += pow(a[i] - b[i], 2);
	}

	return err;
}