#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

/*
    a - 97 
    b - 98
    ... 
    z - 122

    A - 65
    B - 66
    ...
    Z - 90
*/

__global__ void cifrado_cesar(char *texto, int desplazamiento) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (texto[idx] >= 'a' && texto[idx] <= 'z') {
        texto[idx] = (texto[idx] - 'a' + desplazamiento) % 26 + 'a';
    } else if (texto[idx] >= 'A' && texto[idx] <= 'Z') {
        texto[idx] = (texto[idx] - 'A' + desplazamiento) % 26 + 'A';
    }
}

__global__ void descifrado_cesar(char *texto, int desplazamiento) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (texto[idx] >= 'a' && texto[idx] <= 'z') {
        texto[idx] = (texto[idx] - 'a' - desplazamiento + 26) % 26 + 'a';
    } else if (texto[idx] >= 'A' && texto[idx] <= 'Z') {
        texto[idx] = (texto[idx] - 'A' - desplazamiento + 26) % 26 + 'A';
    }
}

void leer_archivo(const char *nombre_archivo, char **contenido) {
    FILE *archivo = fopen(nombre_archivo, "r");
    if (archivo == NULL) {
        perror("Error al abrir el archivo para lectura");
        exit(EXIT_FAILURE);
    }
    
    fseek(archivo, 0, SEEK_END);
    long tamano = ftell(archivo);
    rewind(archivo);
    
    *contenido = (char *)malloc(tamano + 1);
    if (*contenido == NULL) {
        perror("Error al asignar memoria");
        fclose(archivo);
        exit(EXIT_FAILURE);
    }
    
    fread(*contenido, 1, tamano, archivo);
    (*contenido)[tamano] = '\0';
    
    fclose(archivo);
}

void escribir_archivo(const char *nombre_archivo, const char *contenido) {
    FILE *archivo = fopen(nombre_archivo, "w");
    if (archivo == NULL) {
        perror("Error al abrir el archivo para escritura");
        exit(EXIT_FAILURE);
    }
    
    fprintf(archivo, "%s", contenido);
    fclose(archivo);
}

int main() {
    char *contenido = NULL;
    char *contenido_device = NULL;
    //const char *archivo_entrada = "entrada.txt";
    const char *archivo_salida_cifrado = "salida_cifrado.txt";
    const char *archivo_salida_descifrado = "salida_descifrado.txt";
    int nHilos, nBloques;
    
    int desplazamiento, cifrado_descifrado;
    char ruta[100];

    printf("\n\n ------------- CIFRADO CESAR -------------\n");

    // Leer las condiciones del usuario
    printf("Ingrese la ruta del archivo: ");
    scanf("%s", ruta);
    printf("Ingrese el desplazamiento: ");
    scanf("%d", &desplazamiento);
    printf("Ingrese 1 para cifrar o 2 para descifrar: ");
    scanf("%d", &cifrado_descifrado);

    leer_archivo(ruta, &contenido);
    printf("\nContenido leido del txt: <<< %s >>>\n", contenido);
    
    // Reservar memoria en el device para el contenido (usando la longitud del contenido)
    cudaMalloc((void **)&contenido_device, sizeof(char) * strlen(contenido));
    printf("Memoria reservada en el device\n");
    
    // Copiar el contenido al device
    cudaMemcpy(contenido_device, contenido, sizeof(char) * strlen(contenido), cudaMemcpyHostToDevice);
    printf("Contenido copiado al device\n");

    nHilos = 512;
    nBloques = ((strlen(contenido) / nHilos) + 1);
    if(cifrado_descifrado == 1) {
        printf("Cifrando...\n");
        // Llamar a la función cifrado_cesar
        cifrado_cesar<<<nBloques, nHilos>>>(contenido_device, desplazamiento);
        cudaDeviceSynchronize(); // se asegura que el kernel haya terminado de ejecutarse
        cudaMemcpy(contenido, contenido_device, sizeof(char) * strlen(contenido), cudaMemcpyDeviceToHost);
        escribir_archivo(archivo_salida_cifrado, contenido);
        printf("Cifrado completado\n");
    } else if(cifrado_descifrado == 2) {
        printf("Descifrando...\n");
        // Llamar a la función descifrado_cesar
        descifrado_cesar<<<nBloques, nHilos>>>(contenido_device, desplazamiento);
        cudaDeviceSynchronize(); // se asegura que el kernel haya terminado de ejecutarse
        cudaMemcpy(contenido, contenido_device, sizeof(char) * strlen(contenido), cudaMemcpyDeviceToHost);
        escribir_archivo(archivo_salida_descifrado, contenido);
        printf("Descifrado completado\n");
    } else {
        printf("Opcion no valida\n");
        return 0;
    }
    
    printf("\n\nEjecución finalizada\n\n");
    // Liberar memorias
    cudaFree(contenido_device);
    free(contenido);
    return 0;
}
