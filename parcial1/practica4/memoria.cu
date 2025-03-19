/* Programa que transfiere datos desde la memoria del
host a la  memoria del device usando las funciones de
manejo de memoria de CDUA

Compilar: nvcc 03_mem.cu -o 01_mem.x
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N 2000

// Funcion para sumar una constante en el device
__global__ void suma(float *arreglo, int c)
{
    // for (int i = 0; i < N * N; i++)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N * N)
        arreglo[i] += c;
}

// Función para imprimir un arreglo en el host
void imprimir(float *arreglo)
{
    for (int i = N * N - 10; i < N * N; i++)
        printf("Arreglo[%d]: %.2f\n", i, arreglo[i]);
}

// Función principal ejecutada en el host
int main(int argc, char **argv)
{

    // Declaración de variables
    float *m_host;
    float *m_device;

    int nHilos, nBloques;

    // Reservar memoria en el host
    m_host = (float *)malloc(N * N * sizeof(float));

    // Reservar memoria en el device
    cudaMalloc((void **)&m_device, N * N * sizeof(float));

    // Inicializar la matriz

    for (int i = 0; i < N * N; i++)
    {
        m_host[i] = (float)(rand() % 10);
    }

    printf(" -------------- Arreglo original --------------\n");
    imprimir(m_host);

    // Copiar información al device
    cudaMemcpy(m_device, m_host, N * N * sizeof(float), cudaMemcpyHostToDevice);

    nHilos = 512;
    nBloques = ((N * N) / nHilos) + 1;

    // Llamar a la función suma
    suma<<<nBloques, nHilos>>>(m_device, 10);
    cudaMemcpy(m_host, m_device, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf(" -------------- Arreglo con la suma --------------\n");
    imprimir(m_host);

    // Liberar memoria
    cudaFree(m_device);

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
