#include "headers.h"

float f(float x); //TODO Implement f function
float Trap(float local_a, float local_b, int local_n, float h); //TODO Implement trap function

int main(int argc, char* argv[]) {
    int p, my_rank;
    MPI_Status status;
    int tag;
    float a, b;
    unsigned long int n;
    long double h, x, integral;
    unsigned long int i;
    

    // Setup MPI communicator
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    //Initial communication
    tag = 0;
    if (my_rank == 0) {
        printf("Enter a, b and n\n");
        scanf("%f %f %lu",&a,&b,&n);
    }
    if (my_rank == 0) {
        for (i=1; i<p; i++) {
            MPI_Send(&a,1,MPI_FLOAT,i,tag,MPI_COMM_WORLD);
            MPI_Send(&b,1,MPI_FLOAT,i,tag,MPI_COMM_WORLD);
            MPI_Send(&n,1,MPI_INT,i,tag,MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&a,1,MPI_FLOAT,0,tag,MPI_COMM_WORLD,&status);
        MPI_Recv(&b,1,MPI_FLOAT,0,tag,MPI_COMM_WORLD,&status);
        MPI_Recv(&n,1,MPI_INT,0,tag,MPI_COMM_WORLD,&status);

        float local_a, local_b, local_int;
        int local_n,
        h = (b-a)/n;
        local_n = n/p;
        // Start of local integration
        local_a = a+my_rank*local_n*h;
        local_b = local_a+local_n*h;
        local_int = Trap(local_a,local_b,local_n,h);

        float temp;
        tag = 1;
        if (my_rank == 0) {
            integral = local_int;
            for (i=1; i<p; i++){
                MPI_Recv(&temp,1,MPI_FLOAT,i,tag,MPI_COMM_WORLD,&status);
                integral = integral + temp;
            }
            printf("With n=%lu trapezoids we estimate integral",n);
            printf(" from %f to %f: %Lf\n",a,b,integral);
        } else {
            MPI_Send(&local_int,1,MPI_FLOAT,0,tag,MPI_COMM_WORLD);
        }
    }
}