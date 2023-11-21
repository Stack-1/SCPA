#include <stdio.h>

#define f(x) x*x
#define FFLUSH_STDOUT_AUTO setvbuf(stdout, NULL, _IONBF, BUFSIZ); //Disable stdout buffering

/* This code works fine, but to get exact result takes a loooooot of time, try with parameters (1,2,1000000000) -> 8.6666667*/

int main(int argc, char* argv[]) {
    float a, b;
    unsigned long int n;
    long double h, x, integral;
    unsigned long int i;
    

    FFLUSH_STDOUT_AUTO
    printf("Enter a, b and n\n");
    scanf("%f %f %lu",&a,&b,&n);
    
    h=(b-a)/n;
    integral = (f(a)+f(b))/2.0;
    
    for (i=1; i<n; i++){
        x = a+i*h;
        integral = integral + f(x);
    }

    integral *= h;
    printf("With n=%lu trapezoids we estimate integral from %f to %f : %Lf\n", n,a,b,integral);
    return 0;
}