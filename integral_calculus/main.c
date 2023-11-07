#include "headers.h"


/*****************************************************************
* This is a program that wants to approximately compute integral * 
* using trapezoidal method using inter-processes communication   *
*****************************************************************/

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
    MPI_Comm_size(MPI_COMM_WORLD, &p); //Get number of processes

    //Initial communication
    tag = 0;
    if (my_rank == 0) {
        printf("My rank is: %d, please enter a, b and n\n",my_rank);
        scanf("%f %f %lu",&a,&b,&n);
    }
    if (my_rank == 0) {
        for (i=1; i<p; i++) {
            MPI_Send(&a,1,MPI_FLOAT,i,tag,MPI_COMM_WORLD);
            MPI_Send(&b,1,MPI_FLOAT,i,tag,MPI_COMM_WORLD);
            MPI_Send(&n,1,MPI_UNSIGNED_LONG,i,tag,MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&a,1,MPI_FLOAT,0,tag,MPI_COMM_WORLD,&status);
        MPI_Recv(&b,1,MPI_FLOAT,0,tag,MPI_COMM_WORLD,&status);
        MPI_Recv(&n,1,MPI_UNSIGNED_LONG,0,tag,MPI_COMM_WORLD,&status);
        printf("[INFO] My rank is: %d, number arrived are: %f %f %lu\n",my_rank,a,b,n);
    }

    // Start local computation
    float local_a, local_b, local_int;
    unsigned long int local_n;
    unsigned long int r;

    // Divide computation fairly between processes
    r = n % p;
    if (my_rank < r)
    {
        local_n = n / p + 1;
        local_a = a + my_rank * local_n * h;
        printf("[DEBUG] My rank is: %d, local values computed are: %f %f %lu\n",my_rank,local_a,local_b,local_n);
    }
    else
    {
        local_n = n / p + 1;
        local_a = a + r * local_n * h;
        local_n = n / p;
        local_a = local_a + (my_rank - r) * local_n * h;
    }
    local_b = local_a + local_n * h;

    h = (b - a) / n;
    printf("[INFO] My rank is: %d, local values computed are: %f %f %lu\n",my_rank,local_a,local_b,local_n);



    // compute_local_integral(local_a,local_b,local_n,h);
    float temp;
    tag = 1;
    if (my_rank == 0)
    {
        integral = local_int;
        for (i = 1; i < p; i++)
        {
            MPI_Recv(&temp, 1, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
            integral = integral + temp;
        }
        printf("With n=%lu trapezoids we estimate integral", n);
        printf(" from %f to %f: %Lf\n", a, b, integral);
    }
    else
    {
        MPI_Send(&local_int, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}