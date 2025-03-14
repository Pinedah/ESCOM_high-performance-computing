// Identifica el número de bloques y hilos en una malla
// y el número de hilos por bloque

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

//Función a ejecutar en el Device
__global__ void identifica(void)
{
  //Definición de variables   
  int IdHilo;
  int IdBloque;
     
  //Obtener el identificador del hilo   
  IdHilo   =  threadIdx.x;
  //Obtener el identificador del bloque
  IdBloque =  blockIdx.x;

  printf("Hola soy el hilo %d del Bloque %d \n",IdHilo, IdBloque);

  if(IdHilo==0 && IdBloque==0)
  {
     //Obtener el número de hilos por bloque
     printf("====Numero de hilos en bloque %d\n",blockDim.x);
     //Obtener el número de bloques por hilo
     printf("====Numero de bloques en una malla %d\n",gridDim.x);
  }
}

// Función principal ejecutada en el host
int main(int argc, char** argv)
{
    printf("Hola, mundo desde el Host!\n");
  
   identifica<<<1,10>>>();

   cudaDeviceSynchronize();
   cudaDeviceReset();
    
   printf("\nPulsa INTRO para finalizar...");
   char tecla = getchar();
   return 0;
}
///////////////////////////////////////////////////////////////////////////

