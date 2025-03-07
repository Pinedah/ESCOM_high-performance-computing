// Ejercicios 4. Búsqueda del estudiante con la calificación más alta. Dada una lista con 10,000 calificaciones (del 0 al 100) se desea encontrar la calificación más alta. 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_GRADES 10000

int main() {
    int grades[NUM_GRADES];
    int highest_grade = 0;

    // Seed the random number generator
    srand(time(NULL));

    // Generate random grades between 0 and 100
    for (int i = 0; i < NUM_GRADES; i++) {
        grades[i] = rand() % 101;
    }

    // Find the highest grade
    for (int i = 0; i < NUM_GRADES; i++) {
        if (grades[i] > highest_grade) {
            highest_grade = grades[i];
        }
    }

    printf("The highest grade is: %d\n", highest_grade);

    return 0;
}