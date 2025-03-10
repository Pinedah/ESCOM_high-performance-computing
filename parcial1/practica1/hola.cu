#include <iostream>
#include <cuda_runtime.h>

// kernel que ejecutará en el device (GPU)
__global__ void holaMundoGPU() {
    printf("La metamorfosis de Franz Kafka (1915), desde la GPU\n");
}

int main() {
    // Lanzamos el kernel en la GPU
    holaMundoGPU<<<1, 1>>>();

    // Mensaje en la CPU
    std::cout << "<<Cuando Gregor Samsa se despertó una mañana después de un sueño intranquilo, se encontró sobre su cama convertido en un monstruoso insecto>>, desde la CPU" << std::endl;

    // Sincronizamos la GPU para asegurar que se impriman ambos mensajes
    cudaDeviceSynchronize();

    return 0;
}
