#include "sum.h"

void generate_random_array(double *arr){

    for(int i=0;i<VERY_LARGE_SIZE;i++){
        arr[i] = (double)rand()/(double)(RAND_MAX/10);
    }
}