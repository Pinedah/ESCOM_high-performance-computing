#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Kernel que se ejecuta en el device (GPU)
__global__ void holaMundoGPU()
{
    printf("Hola mundo desde el device\n\n");
}

int main()
{

    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error == cudaSuccess)
    {
        // La función se ejecutó correctamente
        printf("\nNúmero de dispositivos CUDA disponibles: %d\n", deviceCount);
    }
    else
        // Ocurrió un error al obtener el número de dispositivos CUDA
        printf("Error al obtener el número de dispositivos CUDA: %s\n", cudaGetErrorString(error));

    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if (error == cudaSuccess)
    {
        // La función se ejecutó correctamente, se pueden acceder a las propiedades del dispositivo
        printf("\nPropiedades del dispostivo\n");
        printf("Nombre del dispositivo (device):                        %s \n", prop.name);
        printf("Arquitectura del dispositivo (version mayor y menor):   %d.%d \n", prop.major, prop.minor);
        printf("Tamaño de la memoria global:                            %.2f GB \n", float(prop.totalGlobalMem) / 1000000000);
        printf("Tamaño de la memoria compartida:                        %.2f KB \n", float(prop.sharedMemPerBlock) / 1000);
        printf("Numero de hilos por warp:                               %d \n", prop.warpSize);
        printf("Numero maximo de hilos por bloque:                      %d \n", prop.maxThreadsPerBlock);
        printf("La dimension maxima de los bloques:                     x: %d - y: %d - z: %d \n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("La dimension maxima de las mallas:                      x: %d - y: %d - z: %d \n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        // ... otras propiedades del dispositivo
    }
    else
    {
        // Ocurrió un error al obtener las propiedades del dispositivo
        printf("Error al obtener las propiedades del dispositivo: %s\n",
               cudaGetErrorString(error));
    }

    // Stream para ejecución asíncrona
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Lanzar el kernel en el stream
    holaMundoGPU<<<1, 1, 0, stream>>>();

    // Mensaje en la CPU (host) ejecutándose en paralelo
    std::cout << "Hola mundo desde el host (CPU)" << std::endl;

    // Sincronizar el stream para asegurar que el kernel finaliza antes de salir
    cudaStreamSynchronize(stream);

    // Destruir el stream
    cudaStreamDestroy(stream);

    return 0;
}
