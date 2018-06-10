
// Let it be.
#define _CRT_SECURE_NO_WARNINGS
#define BLOCK_SIZE 1024
#define NUM_OF_ADDS_PER_THREAD 128
#define WARP_SIZE 32
#define SECTION_SIZE 2048

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

// CUDA Kernel
__global__ void
prefixSumCUDA(int *a, size_t n)
{
	/* Implementation of Naive Hillis and Steele Algorithm*/

	int tId = blockIdx.x * blockDim.x + threadIdx.x;

	int end = ceil(log2((float)n));

	for (int offset = 0; offset < end; offset++) {

		if (tId >= n) continue;

		if (tId >= (1 << offset)) {
			a[tId] += a[tId - (1 << offset)];
		}
	}
}

__global__ void
prefixSumMap(int *a, int *subArraySumArray, size_t n, size_t k)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;

	int startPoint = tId * NUM_OF_ADDS_PER_THREAD;
	int endPoint = startPoint + NUM_OF_ADDS_PER_THREAD;
	
	// Scan sub arrays
	for (int i = startPoint + 1; i < endPoint; i++) {
		
		if (i >= n)
			break;

		a[i] += a[i - 1];
	}


	// Block sizes is the same size of the sub arrays
	// Store final results in the subArraySumArray

	__syncthreads();

	// Copy data t
	if(endPoint < n)
		subArraySumArray[tId] = a[endPoint - 1];
}

__global__ void Brent_Kung_scan_kernel(int *a, size_t n) {
	__shared__ int C[SECTION_SIZE];
	int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		C[threadIdx.x] = a[i];
	if (i + blockDim.x < n)
		C[threadIdx.x + blockDim.x] = a[i + blockDim.x];
	for (unsigned int step = 1; step <= blockDim.x; step *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * 2 * step - 1;
		if (index < SECTION_SIZE) {
			C[index] += C[index - step];
		}
	}
	for (int step = SECTION_SIZE / 4; step > 0; step /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1)*step * 2 - 1;
		if (index + step < SECTION_SIZE) {
			C[index + step] += C[index];
		}
	}
	__syncthreads();
	if (i < n)
		a[i] = C[threadIdx.x];
	if (i + blockDim.x < n)
		a[i + blockDim.x] = C[threadIdx.x + blockDim.x];
}

__global__ void
reduceResultsKernel(int *a, int *subArraySumArray, size_t n, size_t k)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;

	int startPoint = tId * NUM_OF_ADDS_PER_THREAD;
	int endPoint = startPoint + NUM_OF_ADDS_PER_THREAD;

	// Each thread adds the result of the scanned blocks array to input array
	int newStartPoint = startPoint + NUM_OF_ADDS_PER_THREAD;
	int newEndPoint = endPoint + NUM_OF_ADDS_PER_THREAD;

	for (int i = newStartPoint; i < newEndPoint; i++) {
		if (i >= n)
			break;

		a[i] += subArraySumArray[tId];
	}
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

	// print_array(b, n);

	// Compute the serial prefix sum on b
	prefix_sum_serial(b, n);

	// Compute the parallel prefix sum on a
	compute_prefix_sum(a, n);

	// Find the computation error 
	float error = compute_mse(a, b, n);

	printf("Computation error is %lf\n", error);

	// Print array
	// print_array(a, n);

	// Free allocated memory
	free(a);

	system("pause");
	return EXIT_SUCCESS;
}


int compute_prefix_sum(int *a, size_t n) {

	// To be allocated on device memory
	int* d_A;
	int* d_subArraySums;

	// Handles errors
	cudaError_t error;

	if (a == NULL)
	{
		fprintf(stderr, "Failed to allocate host vector a!\n");
		exit(EXIT_FAILURE);
	}

	// Allocate memory on device
	int memSize = n * sizeof(int);

	error = cudaMalloc((void **)&d_A, memSize);

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

	// Execution parameters
	// dim3 gridSize(2, 1, 1);
	// dim3 blockSize = (1024, 1, 1);

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

	// Kernel launch 
	// dim3 gridDimensions(ceil((float)n / BLOCK_SIZE), 1, 1);
	// dim3 blockDimensions(BLOCK_SIZE, 1, 1);

	// Allocate memory for the subArraySums array

	int numOfBlocks = ceil((float)n / BLOCK_SIZE);
	int numOfSubArrays = ceil((float)n / NUM_OF_ADDS_PER_THREAD);

	int subArraySumSize = numOfSubArrays * sizeof(int);

	error = cudaMalloc((void **)&d_subArraySums, subArraySumSize);

	if (error != cudaSuccess) {
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	dim3 gridDimensionsMapKernel1(numOfBlocks, 1, 1);
	dim3 blockDimensionsMapKernel1(BLOCK_SIZE, 1, 1);

	dim3 gridDimensionsMapKernel2(ceil((float)numOfSubArrays/BLOCK_SIZE), 1, 1);
	dim3 blockDimensionsMapKernel2(BLOCK_SIZE, 1, 1);

	// prefixSumCUDA <<< gridDimensions, blockDimensions >> > (d_result, d_A, n);

	prefixSumMap << <gridDimensionsMapKernel1, blockDimensionsMapKernel1 >> > (d_A, d_subArraySums, n, numOfSubArrays);
	Brent_Kung_scan_kernel << <gridDimensionsMapKernel2, blockDimensionsMapKernel2 >> > (d_subArraySums, numOfSubArrays);
	reduceResultsKernel << <gridDimensionsMapKernel1, blockDimensionsMapKernel1 >> > (d_A, d_subArraySums, n, numOfSubArrays);

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