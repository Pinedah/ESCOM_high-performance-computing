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

// Kernel para cifrar con Vigenère
__global__ void cifrado_vigenere(char *texto, char *clave, int texto_len, int clave_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < texto_len && ((texto[idx] >= 'a' && texto[idx] <= 'z') || (texto[idx] >= 'A' && texto[idx] <= 'Z'))) {
        char base = (texto[idx] >= 'a' && texto[idx] <= 'z') ? 'a' : 'A';
        int pi = texto[idx] - base; // Posición de la letra en el alfabeto
        int ki = (clave[idx % clave_len] >= 'a' && clave[idx % clave_len] <= 'z') 
                 ? clave[idx % clave_len] - 'a' 
                 : clave[idx % clave_len] - 'A'; // Clave en minúscula o mayúscula
        texto[idx] = base + (pi + ki) % 26; // Cifrar
    }
}

// Kernel para descifrar con Vigenère
__global__ void descifrado_vigenere(char *texto, char *clave, int texto_len, int clave_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < texto_len && ((texto[idx] >= 'a' && texto[idx] <= 'z') || (texto[idx] >= 'A' && texto[idx] <= 'Z'))) {
        char base = (texto[idx] >= 'a' && texto[idx] <= 'z') ? 'a' : 'A';
        int pi = texto[idx] - base; // Posición de la letra en el alfabeto
        int ki = (clave[idx % clave_len] >= 'a' && clave[idx % clave_len] <= 'z') 
                 ? clave[idx % clave_len] - 'a' 
                 : clave[idx % clave_len] - 'A'; // Clave en minúscula o mayúscula
        texto[idx] = base + (pi - ki + 26) % 26; // Descifrar
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

void cifrar_descifrar(int cifrado_descifrado, int bloques, int hilos, char *contenido, char *contenido_device, char *clave_device, int texto_len, int clave_len) {

    const char *archivo_salida_cifrado = "salida_cifrado.txt";
    const char *archivo_salida_descifrado = "salida_descifrado.txt";

    if(cifrado_descifrado == 1) {
        printf("\n Cifrando...\n");
        cifrado_vigenere<<<bloques, hilos>>>(contenido_device, clave_device, texto_len, clave_len);
        cudaDeviceSynchronize(); // se asegura que el kernel haya terminado de ejecutarse
        printf(" ✔  Cifrado completado.\n");

        cudaMemcpy(contenido, contenido_device, sizeof(char) * texto_len, cudaMemcpyDeviceToHost);

        escribir_archivo(archivo_salida_cifrado, contenido);
        printf("\n → El archivo cifrado se ha guardado en el archivo %s\n", archivo_salida_cifrado);

        printf(" → Puede visualizar el contenido del archivo desde la terminal de la siguiente manera:\n");
        printf("\tLinux:\t\tcat %s\n", archivo_salida_cifrado);
        printf("\tWindows:\ttype %s\n", archivo_salida_cifrado);

    } else if(cifrado_descifrado == 2) {
        printf("\n Descifrando...\n");
        descifrado_vigenere<<<bloques, hilos>>>(contenido_device, clave_device, texto_len, clave_len);
        cudaDeviceSynchronize(); // se asegura que el kernel haya terminado de ejecutarse
        printf(" ✔  Descifrado completado.\n");

        cudaMemcpy(contenido, contenido_device, sizeof(char) * texto_len, cudaMemcpyDeviceToHost);

        escribir_archivo(archivo_salida_descifrado, contenido);
        printf("\n → El archivo descifrado se ha guardado en el archivo %s\n", archivo_salida_descifrado);

        printf(" → Puede visualizar el contenido del archivo desde la terminal de la siguiente manera:\n");
        printf("\tLinux:\t\tcat %s\n", archivo_salida_descifrado);
        printf("\tWindows:\ttype %s\n", archivo_salida_descifrado);

    } else {
        printf("Opción no válida :(\n");
        return;
    }
}

int main() {
    char *contenido = NULL;
    char *contenido_device = NULL;
    char *clave_device = NULL;

    int cifrado_descifrado;
    int nHilos, nBloques;
    char ruta[100];
    char clave[100];

    // Mostrar información del programa
    printf("\n☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆ CIFRADO VIGENÈRE ☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆ \n");
    
    printf("\n El cifrado Vigenère es un método de cifrado por sustitución que utiliza una clave repetitiva.\n");
    printf("\n En este programa se cifrará o descifrará un texto leído de un archivo.\n\n");

    // Leer las condiciones del usuario
    printf(" 1. Ingrese la ruta del archivo a leer: ");
    scanf("%s", ruta);
    printf(" 2. Ingrese la clave para el cifrado/descifrado: ");
    scanf("%s", clave);
    printf(" 3. Ingrese 1 para cifrar o 2 para descifrar: ");
    scanf("%d", &cifrado_descifrado);

    leer_archivo(ruta, &contenido);
    
    int texto_len = strlen(contenido);
    int clave_len = strlen(clave);

    // Reservar memoria en el device para el contenido y la clave
    cudaMalloc((void **)&contenido_device, sizeof(char) * texto_len);
    cudaMalloc((void **)&clave_device, sizeof(char) * clave_len);
    
    // Copiar el contenido y la clave al device
    cudaMemcpy(contenido_device, contenido, sizeof(char) * texto_len, cudaMemcpyHostToDevice);
    cudaMemcpy(clave_device, clave, sizeof(char) * clave_len, cudaMemcpyHostToDevice);

    // Calcular el número de hilos y bloques
    nHilos = 512;
    nBloques = ((texto_len / nHilos) + 1);

    // Cifrar o descifrar
    cifrar_descifrar(cifrado_descifrado, nBloques, nHilos, contenido, contenido_device, clave_device, texto_len, clave_len);

    printf("\n☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆ Ejecución finalizada ☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆\n");
    printf(" → developed by: Panké <3\n\n");
    
    // Liberar memorias
    cudaFree(contenido_device);
    cudaFree(clave_device);
    free(contenido);
    return 0;
}