#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

const int RPT = 64;
const int NUMTHR = 1024;

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
 
// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, float **matrixDiagonal, const char *filename, int *num_rows, int *num_cols, int *num_vals)
{
    int err;
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    // Get number of rows, columns, and non-zero values
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");

    int *row_ptr_t = (int *)malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *)malloc(*num_vals * sizeof(int));
    float *values_t = (float *)malloc(*num_vals * sizeof(float));
    float *matrixDiagonal_t = (float *)malloc(*num_rows * sizeof(float));
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *)malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++)
    {
        row_occurances[i] = 0;
    }

    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        row_occurances[row]++;
    }

    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++)
    {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);

    // Set the file position to the beginning of the file
    rewind(file);

    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++)
    {
        col_ind_t[i] = -1;
    }

    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int i = 0, j = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        row--;
        column--;

        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1)
        {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        if (row == column)
        {
            matrixDiagonal_t[j] = value;
            j++;
        }
        i = 0;
    }
    fclose(file);
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
    *matrixDiagonal = matrixDiagonal_t;
}

// CPU implementation of SYMGS using CSR, DO NOT CHANGE THIS
void symgs_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal)
{

    // forward sweep
    for (int i = 0; i < num_rows; i++)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }

    // backward sweep
    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }
        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }
}


//implementation of the first part of the algorithm
__global__ void forwardSweep(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *y, float *matrixDiagonal)
{
    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    if(row<num_rows){    
        float tmp = x[row];
        const int row_start = row_ptr[row];
        const int row_end = row_ptr[row+1];

        for(int col = row_start; col < row_end; col++){
            tmp -= values[col]*x[col_ind[col]];
        }

        tmp = (tmp + (x[row] * matrixDiagonal[row])) / matrixDiagonal[row];
        y[row] = tmp;
    }
    //printf("fine thread #%d\n", (blockIdx.x*blockDim.x + threadIdx.x));
}

//da pensare come ottimizzare
__global__ void backwardSweep(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *y, float *matrixDiagonal)
{
    const int row1 = num_rows - 1 - (blockIdx.x*blockDim.x + threadIdx.x)*RPT;
    const int rowEnd = row1 + RPT;
    if(row1>=0 && rowEnd<num_rows){
        int row = 0;
        for(row = rowEnd; row>=row1; row--){
            float tmp = y[row];
            const int row_start = row_ptr[row];
            const int row_end = row_ptr[row+1];
            bool process = true;

            for(int col = row_start; col < row_end; col++){
                while(process){
                    if(x[col_ind[col]]!=0 || col_ind[col]<=row){
                    tmp -= values[col]*y[col_ind[col]];
                    process = false;
                    }
                }
                process = true;
            }

            process = true;
            while(process){
                if(row == (num_rows-1) || x[row+1]!=0){
                    tmp += y[row] * matrixDiagonal[row];
                    y[row] = tmp / matrixDiagonal[row];
                    x[row]++;
                    process = false;
                }
            }
        }
    }
    printf("fine thread #%d\n", (blockIdx.x*blockDim.x + threadIdx.x));
}


// GPU implementation of SYMGS using CSR
/**
 * @brief 
 * 
 * @param row_ptr pointer to row starts in col_ind and data + last item = to muber of values
 * @param col_ind array of column indeces
 * @param values array of corresponding nonzero values
 * @param num_rows number of rows
 * @param x vector to multiply for
 * @param matrixDiagonal vector of diagonal values
 * @return __global__ 
 */
/*
__global__ void symgsGPU(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *y, float *matrixDiagonal)
{
    //forwardSweep
    forwardSweep(row_ptr, col_ind, values, num_rows, x, y, matrixDiagonal);
}*/


int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }

    //CPU vaiables
    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    float *matrixDiagonal;

    const char *filename = argv[1];

    double start_time, end_time;

    //lettuta matrice e inizializzazione variabili
    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename, &num_rows, &num_cols, &num_vals);
    float *x = (float *)malloc(num_rows * sizeof(float));
    printf("END reading matrix\n");
    //Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (rand() % 100) / (rand() % 100 + 1); // the number we use to divide cannot be 0, that's the reason of the +1
    }
    
    printf("X generated\n");

    //GPU vaiables
    int *d_row_ptr, *d_col_ind;
    float *d_values;
    float *d_matrixDiagonal;
    float *d_x, *d_y;
 
    //allocazione memoria per vettori su gpu
    cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, num_vals * sizeof(int));
    cudaMalloc(&d_values, num_vals * sizeof(float));
    cudaMalloc(&d_matrixDiagonal, num_rows * sizeof(float));
    cudaMalloc(&d_x, num_rows * sizeof(float));
    cudaMalloc(&d_y, num_rows * sizeof(float));

    //copia e inizializzazione vettori su gpu
    cudaMemcpy(d_row_ptr, row_ptr,  (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind,  num_vals * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values,  num_vals * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixDiagonal, matrixDiagonal,  num_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x,  num_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, num_rows * sizeof(float));

    printf("GPU START\n");
    start_time = get_time();
    dim3 blocksPerGrid(num_rows/(NUMTHR*RPT), 1, 1);
    dim3 ThreadsPerBlock(NUMTHR, 1, 1);
    forwardSweep<<<blocksPerGrid, ThreadsPerBlock>>>(d_row_ptr, d_col_ind, d_values, num_rows, d_x, d_y, d_matrixDiagonal);
    cudaDeviceSynchronize();
    cudaMemset(d_x, 0, num_rows * sizeof(float));
    backwardSweep<<<blocksPerGrid, ThreadsPerBlock>>>(d_row_ptr, d_col_ind, d_values, num_rows, d_x, d_y, d_matrixDiagonal);
    cudaDeviceSynchronize();
    end_time = get_time();
    printf("SYMGS Time GPU: %.10lf\n", end_time - start_time);

    //creo vettore di supporto per il testing e copio il risultato della gpu
    float *y = (float *)malloc(num_rows * sizeof(float));
    cudaMemcpy(y, d_y,  num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_matrixDiagonal);
    cudaFree(d_x);
    cudaFree(d_y);

    // Compute in sw
    start_time = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_time = get_time();

    // Print time
    printf("SYMGS Time CPU: %.10lf\n", end_time - start_time);

    bool correct = true;
    int i = 0;
    for(i = 0; i<num_rows && correct; i++){
        correct = (y[i] == x[i]);
    }

    if(correct){
        printf("Yeeeee\n");
    }else{
        printf("fail\n");
        printf("error at %d. Val x = %f, Val y = %f \n", i-1, x[i-1], y[i-1]);
    }
   
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(y);
    free(x);

    return 0;
}