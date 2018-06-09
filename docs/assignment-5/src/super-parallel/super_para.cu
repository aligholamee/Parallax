
 Let it be.
 #define _CRT_SECURE_NO_WARNINGS
 #define BLOCK_SIZE 1024
 #define GRID_SIZE 64
 #define NUM_OF_ADDS_PER_THREAD 16
 #define WARP_SIZE 32
 
 #include stdlib.h
 #include stdio.h
 #include time.h
 #include math.h
 
  CUDA Runtime
 #include cuda_runtime.h
 #include device_launch_parameters.h
 
 void fill_array(int a, size_t n);
 void prefix_sum_serial(int a, size_t n);
 void print_array(int a, size_t n);
 float compute_mse(int a, int b, int n);
 int compute_prefix_sum(int a, size_t n);
 
 __global__ void
 scanMapKernel(int a, size_t n)
 {
     int tId = blockIdx.x  blockDim.x + threadIdx.x;
     int subStartPoint = tId  NUM_OF_ADDS_PER_THREAD;
     int subEndPoint = subStartPoint  NUM_OF_ADDS_PER_THREAD;
 
      Each thread scans its own subarray
     for (int i = subStartPoint + 1; i  subEndPoint; i++) {
 
         if (i = n)
             break;
 
         a[i] += a[i - 1];
     }
 
      Wait all sub arrays get computed
     __syncthreads();
 
      Define intermediate array (Handles subarrays)
     extern __shared__ int subArraySumArray[];
 
      Load final element of subarrays into intermediate array
      Valid for all threads
     subArraySumArray[threadIdx.x] = a[subEndPoint - 1];
 
      Scan the intermediate array (Separate blocks)
      Done by first thread of each block
     if (threadIdx.x == 0) {
         for (int j = 1; j  (BLOCK_SIZE - 1)NUM_OF_ADDS_PER_THREAD + 1; j++) {
             subArraySumArray[j] += subArraySumArray[j - 1];
         }
     }
 
     __syncthreads();
 
      Define intermediate array (Handles blocks)
     __shared__ int blockSumArray[GRID_SIZE];
 
      Load final element of blocks into intermediate array
      Valid for last thread of the block
     int blockStartPoint = (subStartPoint - 1)  NUM_OF_ADDS_PER_THREAD + 1;
     int blockEndPoint = (subEndPoint - 1)  NUM_OF_ADDS_PER_THREAD + 1;
 
     if (threadIdx.x == BLOCK_SIZE - 1) {
         blockSumArray[blockIdx.x] = subArraySumArray[blockEndPoint];
     }
 
      Scan the intermediate array (Global blocks)
      Done by 1 thread D
     if (tId == 0) {
         for (int j = 1; j  GRID_SIZE; j++) {
             blockSumArray[j] = blockSumArray[j - 1];
         }
     }
 
      Update subArraySumArray
     int bias = (BLOCK_SIZE - 1)  NUM_OF_ADDS_PER_THREAD + 1;
     int newBlockStartPoint = blockStartPoint + bias;
     int newBlockEndPoint = blockEndPoint + bias;
     
     for (int i = newBlockStartPoint; i  newBlockEndPoint; i++) {
         subArraySumArray[i] += blockSumArray[threadIdx.x];
     }
 
      Update the main array
     int newStartPoint = subStartPoint + NUM_OF_ADDS_PER_THREAD;
     int newEndPoint = subEndPoint + NUM_OF_ADDS_PER_THREAD;
 
     for (int i = newStartPoint; i  newEndPoint; i++) {
         if (i = n)
             break;
 
         a[i] += subArraySumArray[tId];
     }
 }
 
 void scanCPU(float f_out, float f_in, int i_n)
 {
     f_out[0] = 0;
     for (int i = 1; i  i_n; i++)
         f_out[i] = f_out[i - 1] + f_in[i - 1];
 }
 
 int main(int argc, char argv[]) {
      Input N
     size_t n = 0;
     printf([-] Please enter N );
     scanf(%uldn, &n);
      Allocate memory for array
     int  a = (int )malloc(n  sizeof a);
      Fill array with numbers 1..n
     fill_array(a, n);
      Print array
      print_array(a, n);
      Compute prefix sum
      prefix_sum(a, n);
 
      Create a copy of a for comparison
     int  b = (int )malloc(n  sizeof b);
     for (int i = 0; i  n; i++) {
         b[i] = a[i];
     }
 
      print_array(b, n);
 
      Compute the serial prefix sum on b
     prefix_sum_serial(b, n);
 
      Compute the parallel prefix sum on a
     compute_prefix_sum(a, n);
 
      Find the computation error 
     float error = compute_mse(a, b, n);
 
     printf(Computation error is %lfn, error);
 
      Print array
      print_array(a, n);
 
      Free allocated memory
     free(a);
 
     system(pause);
     return EXIT_SUCCESS;
 }
 
 
 int compute_prefix_sum(int a, size_t n) {
 
      To be allocated on device memory
     int d_A;
     int d_subArraySums;
 
      Handles errors
     cudaError_t error;
 
     if (a == NULL)
     {
         fprintf(stderr, Failed to allocate host vector a!n);
         exit(EXIT_FAILURE);
     }
 
      Allocate memory on device
     int memSize = n  sizeof(int);
 
     error = cudaMalloc((void )&d_A, memSize);
 
     if (error != cudaSuccess) {
         printf(cudaMalloc d_A returned error %s (code %d), line(%d)n, cudaGetErrorString(error), error, __LINE__);
         exit(EXIT_FAILURE);
     }
 
      Copy data from host to device
     error = cudaMemcpy(d_A, a, n  sizeof(int), cudaMemcpyHostToDevice);
 
     if (error != cudaSuccess) {
         printf(cudaMemcpy d_A returned error %s (code %d), line(%d)n, cudaGetErrorString(error), error, __LINE__);
         exit(EXIT_FAILURE);
     }
 
     printf(Processing data on GPU...n);
 
      Setup CUDA events for timings
     cudaEvent_t start;
 
      Create the CUDA events
     error = cudaEventCreate(&start);
 
     if (error != cudaSuccess)
     {
         fprintf(stderr, Failed to create start event (error code %s)!n, cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
 
     cudaEvent_t stop;
     error = cudaEventCreate(&stop);
 
     if (error != cudaSuccess)
     {
         fprintf(stderr, Failed to create stop event (error code %s)!n, cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
 
      Record the start event
     error = cudaEventRecord(start, NULL);
 
     if (error != cudaSuccess)
     {
         fprintf(stderr, Failed to record record start event (error code %s)!n, cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
 
 
      Allocate memory for the subArraySums array
     int numOfBlocks = ceil((float)n  BLOCK_SIZE);
 
     dim3 gridDimensionsMapKernel(numOfBlocks, 1, 1);
     dim3 blockDimensionsMapKernel(BLOCK_SIZE, 1, 1);
 
     int subArraySumArraySize = ceil((float)n  NUM_OF_ADDS_PER_THREAD);
 
     scanMapKernel  gridDimensionsMapKernel, blockDimensionsMapKernel, subArraySumArraySize   (d_A, n);
 
      Check kernel launch
     error = cudaGetLastError();
     if (error != cudaSuccess)
     {
         fprintf(stderr, Failed to launch kernel!n, cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
 
      Record the stop event
     error = cudaEventRecord(stop, NULL);
 
     if (error != cudaSuccess)
     {
         fprintf(stderr, Failed to record stop event (error code %s)!n, cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
 
 
      Wait for the stop event to be completed by all threads
     error = cudaEventSynchronize(stop);
 
     if (error != cudaSuccess)
     {
         fprintf(stderr, Failed to synchronize on the stop event (error code %s)!n, cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
 
      Compute the elapsed time
     float elapsed_time = 0.0f;
     error = cudaEventElapsedTime(&elapsed_time, start, stop);
 
     printf(Elapsed time in msec = %lfn, elapsed_time);
 
     if (error != cudaSuccess)
     {
         fprintf(stderr, Failed to get time elapsed between events (error code %s)!n, cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
 
      Copy result from device to host
     error = cudaMemcpy(a, d_A, n  sizeof(int), cudaMemcpyDeviceToHost);
 
 
     if (error != cudaSuccess)
     {
         printf(cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)n, cudaGetErrorString(error), error, __LINE__);
         exit(EXIT_FAILURE);
     }
 
     cudaFree(d_A);
 
     return EXIT_SUCCESS;
 }
 void prefix_sum_serial(int a, size_t n) {
     int i;
     for (i = 1; i  n; ++i) {
         a[i] = a[i] + a[i - 1];
     }
 }
 
 void print_array(int a, size_t n) {
     int i;
     printf([-] array );
     for (i = 0; i  n; ++i) {
         printf(%d, , a[i]);
     }
     printf(bb n);
 }
 
 void fill_array(int a, size_t n) {
     int i;
     for (i = 0; i  n; ++i) {
         a[i] = i + 1;
     }
 }
 
 float compute_mse(int a, int b, int n) {
     float err;
 
     for (int i = 0; i  n; i++) {
         err += pow(a[i] - b[i], 2);
     }
 
     return err;
 }