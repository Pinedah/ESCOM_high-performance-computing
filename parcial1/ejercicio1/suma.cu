/* Programa que transfiere datos desde la memoria del
host a la  memoria del device usando las funciones de
manejo de memoria de CDUA

Compilar: nvcc 03_mem.cu -o 01_mem.x
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N 2000
#define M 3000

// Funcion para sumar dos matrices en el device
__global__ void suma(float *m1, float *m2, float *m3)
{
    // for (int i = 0; i < N * N; i++)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N * M)
        m3[i] = m1[i] + m2[i];
}

// Función para imprimir un arreglo en el host
void imprimir(float *arreglo)
{
    for (int i = N - 10; i < N; i++){
        for (int j = M - 10 ; j < M; j++)
            //printf("Arreglo[%d][%d]: %.2f\n", i, j, arreglo[N * i + j]);
            printf("%.1f\t", arreglo[N * i + j]);
        printf("\n");
    }
}

void suma_host(float *m1, float *m2, float *m_res)
{
    for (int i = 0; i < N; i++)
        for(int j = 0; j < M; j++)
            m_res[N * i + j] = m1[N * i + j] + m2[N * i + j];
}

// Función principal ejecutada en el host
int main(int argc, char **argv)
{
    // Declaración de variables
    float *m1_host, *m2_host, *m_res_host;
    float *m1_device, *m2_device, *m_res_device;

    int nHilos, nBloques;

    // Reservar memoria en el host
    m1_host = (float *)malloc(N * M * sizeof(float));
    m2_host = (float *)malloc(N * M * sizeof(float));
    m_res_host = (float *)malloc(N * M * sizeof(float));

    // Reservar memoria en el device
    cudaMalloc((void **)&m1_device, N * M * sizeof(float));
    cudaMalloc((void **)&m2_device, N * M * sizeof(float));
    cudaMalloc((void **)&m_res_device, N * M * sizeof(float));

    // Inicializar la matriz

    for (int i = 0; i < N * M; i++)
    {
        m1_host[i] = (float)(rand() % 10);
        m2_host[i] = (float)(rand() % 10);
    }

    printf(" -------------- Matrices originales --------------\n");
    printf(" -------------- Matriz 1 --------------\n");
    imprimir(m1_host);
    printf(" -------------- Matriz 2 --------------\n");
    imprimir(m2_host);
    
    printf(" -------------- Suma en el Host --------------\n");
    suma_host(m1_host, m2_host, m_res_host);
    imprimir(m_res_host);

    // Copiar información al device
    cudaMemcpy(m1_device, m1_host, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m2_device, m2_host, N * M * sizeof(float), cudaMemcpyHostToDevice);

    nHilos = 512;
    nBloques = ((N * M) / nHilos) + 1;

    // Llamar a la función suma
    suma<<<nBloques, nHilos>>>(m1_device, m2_device, m_res_device);
    cudaMemcpy(m_res_host, m_res_device, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    printf(" -------------- Suma de matrices en el device --------------\n");
    imprimir(m_res_host);

    // Liberar memoria
    cudaFree(m1_device);
    cudaFree(m2_device);
    cudaFree(m_res_device);

    size_t libre, total;
    cudaMemGetInfo(&libre, &total);
    printf("\nMemoria libre: %zu bytes\n", libre);
    printf("Memoria total: %zu bytes\n", total);

    printf("\npulsa INTRO para finalizar...");
    fflush(stdin);
    char tecla = getchar();

    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////
