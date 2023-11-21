all:
	mpicc -o hello_world first_mpi.c
	mpirun -np $(PROCESS_NUMBER) ./hello_world

clean:
	rm hello_world