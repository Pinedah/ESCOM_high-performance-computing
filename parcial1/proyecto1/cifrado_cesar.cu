#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

/*
    a - 97 
    ... 
    z - 122

    A - 65
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
    const char *archivo_salida_cifrado = "salida_cifrado.txt";
    const char *archivo_salida_descifrado = "salida_descifrado.txt";

    int desplazamiento, cifrado_descifrado;
    int nHilos, nBloques;
    char ruta[100];

    // Mostrar información del programa
    printf("\n☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆ CIFRADO CESAR ☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆ \n");
    
    printf("\nEl cifrado cesar es un tipo de cifrado por sustitución en \n");
    printf("el que una letra en el texto original es reemplazada por \n");
    printf("otra letra que se encuentra un número fijo de posiciones \n");
    printf("más adelante en el alfabeto.\n");
    printf("\nEn este programa se cifrará o descifrará un texto leído de un archivo.\n\n");

    // Leer las condiciones del usuario
    printf("1. Ingrese la ruta del archivo a leer: ");
    scanf("%s", ruta);
    printf("2. Ingrese el desplazamiento: ");
    scanf("%d", &desplazamiento);
    printf("3. Ingrese 1 para cifrar o 2 para descifrar: ");
    scanf("%d", &cifrado_descifrado);

    leer_archivo(ruta, &contenido);
    
    // Reservar memoria en el device para el contenido (usando la longitud del contenido)
    cudaMalloc((void **)&contenido_device, sizeof(char) * strlen(contenido));
    
    // Copiar el contenido al device
    cudaMemcpy(contenido_device, contenido, sizeof(char) * strlen(contenido), cudaMemcpyHostToDevice);

    // Calcular el número de hilos y bloques
    nHilos = 512;
    nBloques = ((strlen(contenido) / nHilos) + 1);

    // Cifrar o descifrar el contenido
    if(cifrado_descifrado == 1) {
        printf("\n ▀  Cifrando...\n");
        cifrado_cesar<<<nBloques, nHilos>>>(contenido_device, desplazamiento);
        cudaDeviceSynchronize(); // se asegura que el kernel haya terminado de ejecutarse
        printf(" ✔  Cifrado completado.\n");

    } else if(cifrado_descifrado == 2) {
        printf("\n ▀  Descifrando...\n");
        descifrado_cesar<<<nBloques, nHilos>>>(contenido_device, desplazamiento);
        cudaDeviceSynchronize(); // se asegura que el kernel haya terminado de ejecutarse
        printf(" ✔  Descifrado completado.\n");

    } else {
        printf("Opcion no valida :(\n");
        return 0;
    }

    // Copiar el contenido al host
    cudaMemcpy(contenido, contenido_device, sizeof(char) * strlen(contenido), cudaMemcpyDeviceToHost);

    if(cifrado_descifrado == 1) {
        escribir_archivo(archivo_salida_cifrado, contenido);
        printf("\n-> El archivo cifrado se ha guardado en el archivo %s\n", archivo_salida_cifrado);
        
    } else if(cifrado_descifrado == 2) {
        escribir_archivo(archivo_salida_descifrado, contenido);
        printf("\n-> El archivo descifrado se ha guardado en el archivo %s\n", archivo_salida_descifrado);
    }


    printf("-> Puede visualizar el contenido del archivo desde la terminar de la siguiente manera:\n");
    printf("Usando linux -> cat %s\n", archivo_salida_cifrado);
    printf("Usando windows -> type %s\n", archivo_salida_cifrado);
    printf("\n☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆ Ejecución finalizada ☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆\n\n");
    
    // Liberar memorias
    cudaFree(contenido_device);
    free(contenido);
    return 0;
}
