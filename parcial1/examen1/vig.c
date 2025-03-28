// Función que cifra y descifra el texto mediante el algoritmo de Vigenère
void vigenere(char *input, char *key, int encrypt) {
    int inputLen = strlen(input);
    int keyLen = strlen(key);
    char base;
    int i, j;
    int pi,ki;

    for (i = 0, j = 0; i < inputLen; i++) {
        // Solo cifrar letras
        if (isalpha(input[i])) {
            base = isupper(input[i]) ? 'A' : 'a';
            pi = input[i] - base; // Posición de la letra en el alfabeto (0-25)
            ki = tolower(key[j % keyLen]) - 'a'; // Posición de la letra de la clave en el alfabeto, forzada a minúscula

            // Elegir entre cifrar y descifrar
            if (encrypt) {
                input[i] = base + (pi + ki) % 26;
            } else {
                input[i] = base + (pi - ki + 26) % 26;
            }
        }
        j= j + 1; // Avanzar en la clave
    }
}