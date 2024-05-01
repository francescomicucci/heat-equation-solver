#include <cmath>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include "parallelHeat.hpp"

int main(int argc, char *argv[])
{
    int rank, size;                     // Rank of the process and total number of processes
    int ndims = 3;                      // Number of dimensions of the Cartesian grid
    int dims[ndims] = {0, 0, 0};        // Array specifying the number of processes in each dimension
    int periods[ndims] = {0, 0, 0};     // Array specifying if the grid is periodic in each dimension
    int coord[ndims];                   // Array specifying the Cartesian coordinates of the process
    int nx = 100;                       // Number of points in a generic direction

    auto startTime = std::chrono::high_resolution_clock::now();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);    // Get the number of processes
    MPI_Dims_create(size, ndims, dims);     // Create a division of processes in a Cartesian grid
    
    // Create a Cartesian communicator
    MPI_Comm commCart;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &commCart);

    MPI_Comm_rank(commCart, &rank);                     // Get the rank of the process in the communicator
    MPI_Cart_coords(commCart, rank, ndims, coord);      // Get the coordinates of the process in the communicator

    // Print auxiliary information
    if (rank == 0){
        std::cout << "============================================" << std::endl;
        std::cout << "Dimensions of the Cartesian grid: " << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;
    }

    // Divide domain points in subdomains    
    int nxSubdomain = nx / dims[0];             // Number of points in the x-direction of one of the 1st (dims[0]-1) subdomains
    int nySubdomain = nx / dims[1];             // Number of points in the y-direction of one of the 1st (dims[1]-1) subdomains
    int nzSubdomain = nx / dims[2];             // Number of points in the z-direction of one of the 1st (dims[2]-1) subdomains
    int start[ndims];                           // Array specifying the minimum coordinates of a point in the subdomain
    int end[ndims];                             // Array specifying the maximum coordinates of a point in the subdomain
    int points[ndims] = {nxSubdomain, nySubdomain, nzSubdomain};

    for(int d = 0; d < ndims; d++){
        start[d] = coord[d] * points[d];
        if(coord[d] != dims[d] - 1)
            end[d] = (coord[d] + 1) * points[d] - 1;
        else
            end[d] = nx - 1;
    }

    int localSize[ndims] = {end[0] - start[0] + 1, end[1] - start[1] + 1, end[2] - start[2] + 1};
    
    // Print auxiliary information
    // std::cout << "Process: " << rank + 1 << "/" << size << " with coordinates: " 
    //           << coord[0] << " " << coord[1] << " " << coord[2] 
    //           << ", start = [" << start[0] << " " << start[1] << " " << start[2] << "], end = ["
    //           << end[0] << " " << end[1] << " " << end[2] << "]" << std::endl;

    ParallelHeat heat(rank, size, dims, coord, start, end, localSize, commCart, nx, 100);    
    heat.solve();

    MPI_Finalize();

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    if (rank == 0){
        std::cout << "Time: " << elapsed.count() << " milliseconds " << std::endl;
        std::cout << "============================================" << std::endl;
    }

    return 0;
}