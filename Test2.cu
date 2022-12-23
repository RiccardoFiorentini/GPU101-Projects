#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                                       \
{                                                                                     \
    const cudaError_t err = call;                                                     \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
}
 
#define CHECK_KERNELCALL()                                                                \
{                                                                                     \
    const cudaError_t err = cudaGetLastError();                                       \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
}

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
 
// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, float **matrixDiagonal, const char *filename, int *num_rows, int *num_cols, int *num_vals)
{
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
//    for (int i = 0; i < num_rows; i++)
//    {
//        float sum = x[i];
//        const int row_start = row_ptr[i];
//        const int row_end = row_ptr[i + 1];
//        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value
//
//        for (int j = row_start; j < row_end; j++)
//        {
//            sum -= values[j] * x[col_ind[j]];
//        }
//
//        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop
//
//        x[i] = sum / currentDiagonal;
//    }

    //backward sweep
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

//implementation of the forward sweep single iteration, it compute all the line that are indipendent at it's iterated in the main
__global__ void forwardSweep(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *y, int *modified, float *matrixDiagonal, int* finish)
{ 
    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    if((row<num_rows) && modified[row]==0){
        float tmp = x[row];
        const int row_start = row_ptr[row];
        const int row_end = row_ptr[row+1];
        bool done = true;
        for(int col = row_start; (col < row_end) && done; col++){
            if(col_ind[col]>=row){
                tmp -= values[col]*x[col_ind[col]];
            }else if(modified[col_ind[col]]==1){
                tmp -= values[col]*y[col_ind[col]];
            }else if(modified[col_ind[col]]==0){
                    done = false;    
            }
        }
        if(done){
            tmp += x[row] * matrixDiagonal[row];
            y[row] = tmp / matrixDiagonal[row];
            modified[row] = 1;
        }else{
            finish[0] = 0;
        }
   }
    __syncthreads();   
}

//Non ancora implementato in modo generico, funziona solo per triangolari inferiori
__global__ void backwardSweep(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *y, float *matrixDiagonal, int *modified, int *finish)
{
    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    if((row<num_rows) && modified[row]==0){
        float tmp = x[row];
        const int row_start = row_ptr[row];
        const int row_end = row_ptr[row+1];
        bool done = true;
        for(int col = row_start; (col < row_end) && done; col++){
            if(col_ind[col]<=row){
                tmp -= values[col]*x[col_ind[col]];
            }else if(modified[col_ind[col]]==1){
                tmp -= values[col]*y[col_ind[col]];
            }else if(modified[col_ind[col]]==0){
                    done = false;    
            }
        }
        if(done){
            tmp += x[row] * matrixDiagonal[row];
            y[row] = tmp / matrixDiagonal[row];
            modified[row] = 1;
        }else{
            finish[0] = 0;
        }
   }

    //const int row = blockIdx.x*blockDim.x + threadIdx.x;
    //if(row<num_rows){
    //    float tmp = x[row];
    //    const int row_start = row_ptr[row];
    //    const int row_end = row_ptr[row+1];
    //    for(int col = row_start; col < row_end; col++){
    //        tmp -= values[col]*x[col_ind[col]];
    //    }
    //    tmp += x[row] * matrixDiagonal[row];
    //    y[row] = tmp / matrixDiagonal[row];
    //}
    __syncthreads();
}

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
    int finish = 0;

    const char *filename = argv[1];

    double start_time, end_time;

    //lettuta matrice e inizializzazione variabili
    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename, &num_rows, &num_cols, &num_vals);
    float *x = (float *)malloc(num_rows * sizeof(float));
    printf("END reading matrix\n");

    col_ind[num_vals-1] = num_rows - 1; 

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
    int *d_modified;
    int *d_finish;

    const int setter = 1;

    CHECK(cudaMalloc(&d_finish, sizeof(int)));
    CHECK(cudaMemcpy(d_finish, &setter, sizeof(int), cudaMemcpyHostToDevice));
    //CHECK(cudaMemset((int *)d_finish, 1, sizeof(int)));//set che non funziona -------------------
 
    //allocazione memoria per vettori su gpu
    CHECK(cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_col_ind, num_vals * sizeof(int)));
    CHECK(cudaMalloc(&d_values, num_vals * sizeof(float)));
    CHECK(cudaMalloc(&d_matrixDiagonal, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&d_x, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&d_y, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&d_modified, num_rows * sizeof(int)));

    //copia e inizializzazione vettori su gpu
    CHECK(cudaMemset(d_modified, false, num_rows * sizeof(int)));
    CHECK(cudaMemcpy(d_row_ptr, row_ptr,  (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col_ind, col_ind,  num_vals * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values,  num_vals * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_matrixDiagonal, matrixDiagonal,  num_rows * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, x,  num_rows * sizeof(float), cudaMemcpyHostToDevice));

    //int countCicle = 0;
    //int *checker = (int *)malloc(num_rows * sizeof(int));
    printf("GPU START\n");
    start_time = get_time();
    dim3 blocksPerGrid(ceil(num_rows/1024)+1, 1, 1);
    dim3 ThreadsPerBlock(1024, 1, 1);
    //int j = 1;
    //bool lock = false;
    //while(finish == 0){
    //    CHECK(cudaMemcpy(d_finish, &setter, sizeof(int), cudaMemcpyHostToDevice));
    //    forwardSweep<<<blocksPerGrid, ThreadsPerBlock>>>(d_row_ptr, d_col_ind, d_values, num_rows, d_x, d_y, d_modified, d_matrixDiagonal, d_finish);
    //    cudaDeviceSynchronize();
    //    CHECK_KERNELCALL();
    //    CHECK(cudaMemcpy(&finish, d_finish, sizeof(int), cudaMemcpyDeviceToHost));
    //    //countCicle++;
    //    //printf("end forwardSweep #%d, with finish = %d\n", countCicle, finish); //stampa di controllo per vedere quante volte è stato chiamato il kernel
    //    //CHECK(cudaMemcpy(checker, d_modified, sizeof(int)*num_rows, cudaMemcpyDeviceToHost));
    //    //Codice di controllo per vedere quante righe sono state risolte
    //    //j = 0;
    //    //for(int i=0; i<num_rows; i++){
    //    //    if(checker[i] != 1){
    //    //        j++;
    //    //        if(lock){
    //    //            printf("riga rimanente = %d su un totale di #%d\n", i, num_rows);
    //    //        }
    //    //    }
    //    //}
    //    //printf("rimangono #%d righe da eseguire\n", j);
    //    //if(j == 1){
    //    //    lock = true;
    //    //}
    //}
    //printf("start backward sweep\n");
    
    backwardSweep<<<blocksPerGrid, ThreadsPerBlock>>>(d_row_ptr, d_col_ind, d_values, num_rows, d_x, d_y, d_matrixDiagonal, d_modified, d_finish);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL();
        

    //finish = 0;
    //while(finish == 0){
    //    CHECK(cudaMemcpy(d_finish, &setter, sizeof(int), cudaMemcpyHostToDevice));
    //    backwardSweep<<<blocksPerGrid, ThreadsPerBlock>>>(d_row_ptr, d_col_ind, d_values, num_rows, d_x, d_y, d_matrixDiagonal, d_modified, d_finish);
    //    cudaDeviceSynchronize();
    //    CHECK_KERNELCALL();
    //    CHECK(cudaMemcpy(&finish, d_finish, sizeof(int), cudaMemcpyDeviceToHost));
    //}
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
    cudaFree(d_modified);
    cudaFree(d_finish);

    // Compute in sw
    start_time = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_time = get_time();

    // Print time
    printf("SYMGS Time CPU: %.10lf\n", end_time - start_time);

    //controllo numero di errori
    int count = 0;
    float maxOffset = 0;
    float valAssoc = 0;
    float relativeOff = 0;
    for(int i = 0; i<num_rows; i++){
        if(fabs(y[i] - x[i]) > 0.0001 && fabs((y[i] - x[i])/x[i]) > 0.001){
            count++;
            if(maxOffset<fabs(y[i] - x[i])){
                maxOffset = y[i] - x[i];
                valAssoc = x[i];
                relativeOff = (y[i] - x[i])/x[i];
            }
            //printf("Error line %d with offset = %f it has #%d values on the line and values of the row = %f\n", i, y[i] - x[i], row_ptr[i+1] - row_ptr[i], x[i]);
        }
    }

    printf("Num Errors = %d with max erro = %f with value of %f and relative error of %f\n", count, maxOffset, valAssoc, relativeOff);
   
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(y);
    free(x);

    return 0;
}