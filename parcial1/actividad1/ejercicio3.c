
// Ejercicio 3 : Suma de elementos en una lista. Dada una lista de M números (por ejemplo 10,000 elementos) enteros se desea encontrar la suma total de todos esos elementos.  

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 10000

int main(){
    int lista[M];
    int suma = 0;
    srand(time(NULL));
    for(int i = 0; i < M; i++){
        lista[i] = rand() % 100;
        suma += lista[i];
    }
    printf("La suma de los elementos de la lista es: %d\n", suma);
    return 0;
}