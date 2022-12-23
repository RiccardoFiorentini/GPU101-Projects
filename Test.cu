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

__global__ void kernel(int* set){
    set[0] = 88;
}

__global__ void kernel2(int* arr){
    const int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id<56000000){
        arr[id] = 8;
        //arr[id+(56000000/2)] = 8;
    }
    if(id == 0){
        printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
    }
}


int main(int argc, const char *argv[])
{
    int finish = 0;
    const int setter = 1;

    int *d_finish;

    CHECK(cudaMalloc(&d_finish, sizeof(int)));
    CHECK(cudaMemcpy(d_finish, &setter, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(&finish, d_finish, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Test finish dovrebbe essere = 1 invece è = %d\n", finish);
    dim3 blocksPerGrid(ceil(56000000/1024)+1, 1, 1);
    dim3 ThreadsPerBlock(1024, 1, 1);
    kernel<<<blocksPerGrid, ThreadsPerBlock>>>(d_finish);
    CHECK_KERNELCALL();
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(&finish, d_finish, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Test finish dovrebbe essere = 88 invece è = %d\n", finish);

    int *d_array;
    CHECK(cudaMalloc(&d_array, 56000000 * sizeof(int)));
    int *array = (int *)malloc(sizeof(int) * 56000000);
    
    //dim3 blocksPerGrid1(56000000/(2*1024), 1, 1);

    kernel2<<<blocksPerGrid, ThreadsPerBlock>>>(d_array);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL();
    CHECK(cudaMemcpy(array, d_array, 56000000 * sizeof(int), cudaMemcpyDeviceToHost));
    int count = 0;
    for(int i = 0; i<56000000; i++){
        if(array[i]!=8){
            count++;
            printf("%d -", i);

        }
    }
    printf("tot err = %d\n", count);
    return 0;
}