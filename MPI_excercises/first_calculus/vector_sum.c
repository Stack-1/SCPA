#include "sum.h"

void foo(double *x, double *y, double *z)
{
    for (int i=0; i<VERY_LARGE_SIZE; i++){ 
        z[i] = x[i] + y[i];
        printf("%f ",z[i]);
    }
    puts("");
}

int main(int argc, char **argv){
    double x[VERY_LARGE_SIZE];
    double y[VERY_LARGE_SIZE];
    double z[VERY_LARGE_SIZE];

    generate_random_array(x);
    generate_random_array(y);
    
    foo(x,y,z);

    return 0;
}