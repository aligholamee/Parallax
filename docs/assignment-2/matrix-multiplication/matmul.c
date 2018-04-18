/*
*				In His Exalted Name
*	Title:	Matrix Multiplication Sequential Code
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	15/11/2015
*/

/*
*				Parallelization: Ali Gholami
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS
#define NUM_THREADS 1

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

typedef struct {
	int *A, *B, *C;
	int n, m, p;
} DataSet;

void fillDataSet(DataSet *dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
void multiply(DataSet dataSet);

int main(int argc, char *argv[]){
	DataSet dataSet;
	if(argc < 4){
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <n> <m> <p>\n");
		printf(">>> ");
		scanf("%d %d %d", &dataSet.n, &dataSet.m, &dataSet.p);
	}else{
		dataSet.n = atoi(argv[1]);
		dataSet.m = atoi(argv[2]);
		dataSet.p = atoi(argv[3]);
	}


	// Get the current time 
	float start_time, elapsed_time;

	// Repeat the process 6 times
	int i = 0;
	for(i = 0; i < 6; i++) {
		
		start_time = omp_get_wtime();
		omp_set_num_threads(NUM_THREADS);

		fillDataSet(&dataSet);
		multiply(dataSet);
		printDataSet(dataSet);
		closeDataSet(dataSet);

		elapsed_time += omp_get_wtime() - start_time;
	}

	// report elapsed time
	printf("Average Time Elapsed %lf Secs\n",
	elapsed_time/6 * 10);

	system("PAUSE");
	return EXIT_SUCCESS;
}

void fillDataSet(DataSet *dataSet){
	int i, j;
	
	dataSet->A = (int *) malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->B = (int *) malloc(sizeof(int) * dataSet->m * dataSet->p);
	dataSet->C = (int *) malloc(sizeof(int) * dataSet->n * dataSet->p);
	
	srand(time(NULL));

	for(i = 0; i < dataSet->n; i++){
		for(j = 0; j < dataSet->m; j++){
			dataSet->A[i*dataSet->m + j] = rand() % 100;
		}
	}

	for(i = 0; i < dataSet->m; i++){
		for(j = 0; j < dataSet->p; j++){
			dataSet->B[i*dataSet->p + j] = rand() % 100;
		}
	}
}

void printDataSet(DataSet dataSet){
	int i, j;

	printf("[-] Matrix A\n");
	for(i = 0; i < dataSet.n; i++){
		for(j = 0; j < dataSet.m; j++){
			printf("%-4d", dataSet.A[i*dataSet.m + j]);
		}
		putchar('\n');
	}
	
	printf("[-] Matrix B\n");
	for(i = 0; i < dataSet.m; i++){
		for(j = 0; j < dataSet.p; j++){
			printf("%-4d", dataSet.B[i*dataSet.p + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix C\n");
	for(i = 0; i < dataSet.n; i++){
		for(j = 0; j < dataSet.p; j++){
			printf("%-8d", dataSet.C[i*dataSet.p + j]);
		}
		putchar('\n');
	}
}

void closeDataSet(DataSet dataSet){
	free(dataSet.A);
	free(dataSet.B);
	free(dataSet.C);
}

void multiply(DataSet dataSet){
	int i, j, k, sum;

	#pragma omp parallel for private(i)
	for(i = 0; i < dataSet.n; i++){

		#pragma omp parallel for private(j)
		for(j = 0; j < dataSet.p; j++){
			sum = 0;
			for(k = 0; k < dataSet.m; k++){
				sum += dataSet.A[i * dataSet.m + k] * dataSet.B[k * dataSet.p + j];
			}
			dataSet.C[i * dataSet.p + j] = sum;
		}
	}
}
