// Buscar palabras. En una pila de M textos (M es un número grande, por ejemplo 1,000) se desea contar cuántas veces aparece una palabra clave en ellos.  

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_TEXTS 1000
#define TEXT_LENGTH 1000
#define KEYWORD "example"

int count_keyword_occurrences(char *text, const char *keyword) {
    int count = 0;
    char *pos = text;
    while ((pos = strstr(pos, keyword)) != NULL) {
        count++;
        pos += strlen(keyword);
    }
    return count;
}

int main() {
    char texts[NUM_TEXTS][TEXT_LENGTH];
    int total_count = 0;
    for (int i = 0; i < NUM_TEXTS; i++) 
        snprintf(texts[i], TEXT_LENGTH, "This is an example text. This example is just for testing.");
    for (int i = 0; i < NUM_TEXTS; i++) 
        total_count += count_keyword_occurrences(texts[i], KEYWORD);
    printf("The keyword '%s' appears %d times in the texts.\n", KEYWORD, total_count);
    return 0;
}