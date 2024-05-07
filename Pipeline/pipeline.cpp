#include <array>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h>

using std::array;
using std::vector;
using std::sin;
using std::cos;

const double X = 1.0;              // Length of the domain in each direction
const double T = 1.0;              // Total time of the simulation
const int n = 100;
const int nt = 100;
const int N = n * n * n;
const double dx = X / (n + 1);     // Spacial step
const double dt = T / nt;          // Temporal step

const double diagElem = 1.0 + dt / (dx * dx);
const double offDiagElem = - dt / (2 * dx * dx);

MPI_Comm commCart;
int rank, size;                     // Rank of the process and total number of processes
const int ndims = 3;                // Number of dimensions of the Cartesian grid
int dims[ndims] = {0, 0, 0};        // Array specifying the number of processes in each dimension
int periods[ndims] = {0, 0, 0};     // Array specifying if the grid is periodic in each dimension
int coord[ndims];                   // Array specifying the Cartesian coordinates of the process

array<int, ndims> start;            // Array specifying the minimum coordinates of a point in the subdomain
array<int, ndims> end;              // Array specifying the maximum coordinates of a point in the subdomain
array<int, ndims> points;
array<int, ndims> localSize;        // Array specifying the number of points in the subdomain in each direction

vector<double> localSol;            // Local solution of the linear system
vector<double> deltaSol;

array<double, n> diag;

// Returns the matrix global index associated with a certain mesh point
int getGlobalIndex(int i, int j, int k){
    return i + j * n + k * n * n;
}

// Force function
double fFunction(int x, int y, int z, double t){
    return sin((x + 1) * dx) * sin((y + 1) * dx) * sin((z + 1) * dx) * (3.0 * sin(t) + cos(t));
}

void init(){
    int size = (localSize[0] + 2) * (localSize[1] + 2) * (localSize[2] + 2);

    localSol.resize(size);
    deltaSol.resize(size);

    for(int i = 0; i < size; i++){
        localSol[i] = 0.0;
        deltaSol[i] = 0.0;
    }

    diag[0] = diagElem;
    for(int i = 1; i < n; i++){
        diag[i] = diagElem - offDiagElem * offDiagElem / diag[i - 1];
    }
}

void thomas(const int direction){
    int index;
    int rank_source, rank_prev, rank_next;
    const int size0 = localSize[(direction + 0) % 3] + 2;
    const int size1 = localSize[(direction + 1) % 3] + 2;
    const int start0 = start[(direction + 0) % 3];
    const int start1 = start[(direction + 1) % 3];
    const int start2 = start[(direction + 2) % 3];
    const int end0 = end[(direction + 0) % 3];
    const int end1 = end[(direction + 1) % 3];
    const int end2 = end[(direction + 2) % 3];

    MPI_Cart_shift(commCart, direction, -1, &rank_source, &rank_prev);
    MPI_Cart_shift(commCart, direction, 1, &rank_source, &rank_next);

    // Forward sweep
    for(int k = start2; k <= end2; k++){
        for(int j = start1; j <= end1; j++){
            index = (j - start1 + 1) * size0 + (k - start2 + 1) * size0 * size1;
            
            if(coord[direction] != 0)
                MPI_Recv(&deltaSol[index], 1, MPI_DOUBLE, rank_prev, getGlobalIndex(start0, j, k), commCart, MPI_STATUS_IGNORE);

            for(int i = start0; i <= end0; i++){
                if(i != 0) deltaSol[index + i - start0 + 1] -= (offDiagElem * deltaSol[index + i - start0]) / diag[i - 1];
            }

            if(coord[direction] != dims[direction] - 1)
                MPI_Send(&deltaSol[index + end0 - start0 + 1], 1, MPI_DOUBLE, rank_next, getGlobalIndex(end0 + 1, j, k), commCart);
        }
    }

    // Backward sweep
    for(int k = start2; k <= end2; k++){
        for(int j = start1; j <= end1; j++){
            index = (j - start1 + 1) * size0 + (k - start2 + 1) * size0 * size1;
            if(coord[direction] == dims[direction] - 1)
                deltaSol[index + end0 - start0 + 1] /= diag[n - 1];
            else
                MPI_Recv(&deltaSol[index + end0 - start0 + 2], 1, MPI_DOUBLE, rank_next, getGlobalIndex(end0, j, k), commCart, MPI_STATUS_IGNORE);

            for(int i = end0; i >= start0; i--){
                if(i != n - 1)
                    deltaSol[index + i - start0 + 1] = (deltaSol[index + i - start0 + 1] - offDiagElem * deltaSol[index + i - start0 + 2]) / diag[i];
            }

            if(coord[direction] != 0)
                MPI_Send(&deltaSol[index + 1], 1, MPI_DOUBLE, rank_prev, getGlobalIndex(start0 - 1, j, k), commCart);
        }
    }
}

void rhs(double time){
    int index;
    const int sizeX = localSize[0] + 2;
    const int sizeY = localSize[1] + 2;
    const double coeff = dt / (dx * dx);

    for(int i = start[0]; i <= end[0]; i++){
        for(int j = start[1]; j <= end[1]; j++){
            for(int k = start[2]; k <= end[2]; k++){
                index = (i - start[0] + 1) + (j - start[1] + 1) * sizeX 
                      + (k - start[2] + 1) * sizeX * sizeY;

                deltaSol[index] = dt * (fFunction(i, j, k, time + dt / 2));
                deltaSol[index] -= 6.0 * coeff * localSol[index];

                if(i > 0) deltaSol[index] += coeff * localSol[index - 1];
                if(i < n - 1) deltaSol[index] += coeff * localSol[index + 1];
                else deltaSol[index] += coeff * sin(1.0) * sin((j + 1) * dx) * sin((k + 1) * dx) * sin(time);

                if(j > 0) deltaSol[index] += coeff * localSol[index - sizeX];
                if(j < n - 1) deltaSol[index] += coeff * localSol[index + sizeX];
                else deltaSol[index] += coeff * sin((i + 1) * dx) * sin(1.0) * sin((k + 1) * dx) * sin(time);

                if(k > 0) deltaSol[index] += coeff * localSol[index - sizeX * sizeY];
                if(k < n - 1) deltaSol[index] += coeff * localSol[index + sizeX * sizeY];
                else deltaSol[index] += coeff * sin((i + 1) * dx) * sin((j + 1) * dx) * sin(1.0) * sin(time);
            }
        }
    }
}

void xDirectionSolver(double time){
    int index;
    int sizeX = localSize[0] + 2;
    int sizeY = localSize[1] + 2;

    // BCs
    if(end[0] == n - 1){
        for(int k = start[2]; k <= end[2]; k++){
            for(int j = start[1]; j <= end[1]; j++){
                index = (n - 1 - start[0] + 1) + (j - start[1] + 1) * sizeX 
                      + (k - start[2] + 1) * sizeX * sizeY;
                deltaSol[index] -= offDiagElem * sin(1.0) * sin((j + 1) * dx) * sin((k + 1) * dx) * (sin(time + dt) - sin(time));              
            }
        }
    }
    
    thomas(0);
}

void yDirectionSolver(double time){
    int index;
    int sizeY = localSize[1] + 2;
    int sizeZ = localSize[2] + 2;

    // BCs
    if(end[1] == n - 1){
        for(int k = start[2]; k <= end[2]; k++){
            for(int i = start[0]; i <= end[0]; i++){
                index = (n - 1 - start[1] + 1) + (k - start[2] + 1) * sizeY 
                      + (i - start[0] + 1) * sizeY * sizeZ;
                deltaSol[index] -= offDiagElem * sin((i + 1) * dx) * sin(1.0) * sin((k + 1) * dx) * (sin(time + dt) - sin(time));
            }
        }
    }

    thomas(1);
}

void zDirectionSolver(double time){
    int index;
    int sizeX = localSize[0] + 2;
    int sizeZ = localSize[2] + 2;

    // BCs
    if(end[2] == n - 1){
        for(int j = start[1]; j <= end[1]; j++){
            for(int i = start[0]; i <= end[0]; i++){
                index = (n - 1 - start[2] + 1) + (i - start[0] + 1) * sizeZ 
                      + (j - start[1] + 1) * sizeZ * sizeX;
                deltaSol[index] -= offDiagElem * sin((i + 1) * dx) * sin((j + 1) * dx) * sin(1.0) * (sin(time + dt) - sin(time));
            }
        }
    }

    thomas(2);
}

void rotate(const int direction){
    const int size0 = localSize[(direction + 0) % 3] + 2;
    const int size1 = localSize[(direction + 1) % 3] + 2;
    const int size2 = localSize[(direction + 2) % 3] + 2;

    vector<double> temp(size0 * size1 * size2);

    for(int i = 0; i < size0 * size1 * size2; i++)
        temp[i] = deltaSol[i];
    
    for(int i = 1; i <= localSize[(direction + 0) % 3]; i++){
        for(int j = 1; j <= localSize[(direction + 2) % 3]; j++){
            for(int k = 1; k <= localSize[(direction + 1) % 3]; k++){
                deltaSol[i * size2 * size1 + j * size1 + k] = temp[j * size1 * size0 + k * size0 + i];
            }
        }
    }
}

double finalize(const int t){
    int index;
    const int sizeX = localSize[0] + 2;
    const int sizeY = localSize[1] + 2;
    const int sizeZ = localSize[2] + 2;

    double error = 0.0;
    double errorT = 0.0;
    
    for(int i = 0; i < sizeX * sizeY * sizeZ; i++){
        localSol[i] += deltaSol[i];
    }

    if(t == nt - 1){
        for(int i = start[0]; i <= end[0]; i++){
            for(int j = start[1]; j <= end[1]; j++){
                for(int k = start[2]; k <= end[2]; k++){
                    index = (i - start[0] + 1) + (j - start[1] + 1) * sizeX 
                          + (k - start[2] + 1) * sizeX * sizeY;
                    error += std::pow(localSol[index] - sin((i + 1) * dx) * sin((j + 1) * dx) * sin((k + 1) * dx) * sin(nt * dt), 2);
                }
            }
        }
        MPI_Reduce(&error, &errorT, 1, MPI_DOUBLE, MPI_SUM, 0, commCart);
        errorT = std::sqrt(errorT) / N;
    }

    return errorT;
}

void comunication(){
    const int sizeX = localSize[0] + 2;
    const int sizeY = localSize[1] + 2;
    
    int rank_source, rank_prev, rank_next;

    MPI_Cart_shift(commCart, 0, -1, &rank_source, &rank_prev);
    MPI_Cart_shift(commCart, 0, 1, &rank_source, &rank_next);

    vector<double> send(localSize[1] * localSize[2]);
    vector<double> recv(localSize[1] * localSize[2]);

    // Send to prev on x axis
    for(int i = 1; i <= localSize[2]; i++){
        for(int j = 1; j <= localSize[1]; j++){
            send[(i - 1) * localSize[1] + (j - 1)] = localSol[i * sizeY * sizeX + j * sizeX + 1];
        }
    }
    MPI_Sendrecv(send.data(), localSize[1] * localSize[2], MPI_DOUBLE, rank_prev, 0, recv.data(), localSize[1] * localSize[2], MPI_DOUBLE, rank_next, 0, commCart, MPI_STATUS_IGNORE);
    for(int i = 1; i <= localSize[2]; i++){
        for(int j = 1; j <= localSize[1]; j++){
            localSol[i * sizeY * sizeX + j * sizeX + localSize[0] + 1] = recv[(i - 1) * localSize[1] + (j - 1)];
        }
    }

    // Send to next on x axis
    for(int i = 1; i <= localSize[2]; i++){
        for(int j = 1; j <= localSize[1]; j++){
            send[(i - 1) * localSize[1] + (j - 1)] = localSol[i * sizeY * sizeX + j * sizeX + localSize[0]];
        }
    }
    MPI_Sendrecv(send.data(), localSize[1] * localSize[2], MPI_DOUBLE, rank_next, 0, recv.data(), localSize[1] * localSize[2], MPI_DOUBLE, rank_prev, 0, commCart, MPI_STATUS_IGNORE);
    for(int i = 1; i <= localSize[2]; i++){
        for(int j = 1; j <= localSize[1]; j++){
            localSol[i * sizeY * sizeX + j * sizeX] = recv[(i - 1) * localSize[1] + (j - 1)];
        }
    }

    MPI_Cart_shift(commCart, 1, -1, &rank_source, &rank_prev);
    MPI_Cart_shift(commCart, 1, 1, &rank_source, &rank_next);

    send.resize(localSize[0] * localSize[2]);
    recv.resize(localSize[0] * localSize[2]);

    // Send to prev on y axis
    for(int i = 1; i <= localSize[2]; i++){
        for(int k = 1; k <= localSize[0]; k++){
            send[(i - 1) * localSize[0] + (k - 1)] = localSol[i * sizeY * sizeX + 1 * sizeX + k];
        }
    }
    MPI_Sendrecv(send.data(), localSize[2] * localSize[0], MPI_DOUBLE, rank_prev, 0, recv.data(), localSize[2] * localSize[0], MPI_DOUBLE, rank_next, 0, commCart, MPI_STATUS_IGNORE);
    for(int i = 1; i <= localSize[2]; i++){
        for(int k = 1; k <= localSize[0]; k++){
            localSol[i * sizeY * sizeX + (localSize[1] + 1) * sizeX + k] = recv[(i - 1) * localSize[0] + (k - 1)];
        }
    }

    // Send to next on y axis
    for(int i = 1; i <= localSize[2]; i++){
        for(int k = 1; k <= localSize[0]; k++){
            send[(i - 1) * localSize[0] + (k - 1)] = localSol[i * sizeY * sizeX + localSize[1] * sizeX + k];
        }
    }
    MPI_Sendrecv(send.data(), localSize[2] * localSize[0], MPI_DOUBLE, rank_next, 0, recv.data(), localSize[2] * localSize[0], MPI_DOUBLE, rank_prev, 0, commCart, MPI_STATUS_IGNORE);
    for(int i = 1; i <= localSize[2]; i++){
        for(int k = 1; k <= localSize[0]; k++){
            localSol[i * sizeY * sizeX + k] = recv[(i - 1) * localSize[0] + (k - 1)];
        }
    }

    MPI_Cart_shift(commCart, 2, -1, &rank_source, &rank_prev);
    MPI_Cart_shift(commCart, 2, 1, &rank_source, &rank_next);

    send.resize(localSize[0] * localSize[1]);
    recv.resize(localSize[0] * localSize[1]);

    // Send to prev on z axis
    for(int j = 1; j <= localSize[1]; j++){
        for(int k = 1; k <= localSize[0]; k++){
            send[(j - 1) * localSize[0] + (k - 1)] = localSol[1 * sizeY * sizeX + j * sizeX + k];
        }
    }
    MPI_Sendrecv(send.data(), localSize[0] * localSize[1], MPI_DOUBLE, rank_prev, 0, recv.data(), localSize[0] * localSize[1], MPI_DOUBLE, rank_next, 0, commCart, MPI_STATUS_IGNORE);
    for(int j = 1; j <= localSize[1]; j++){
        for(int k = 1; k <= localSize[0]; k++){
            localSol[(localSize[2] + 1) * sizeY * sizeX + j * sizeX + k] = recv[(j - 1) * localSize[0] + (k - 1)];
        }
    }

    // Send to next on z axis
    for(int j = 1; j <= localSize[1]; j++){
        for(int k = 1; k <= localSize[0]; k++){
            send[(j - 1) * localSize[0] + (k - 1)] = localSol[localSize[2] * sizeY * sizeX + j * sizeX + k];
        }
    }
    MPI_Sendrecv(send.data(), localSize[0] * localSize[1], MPI_DOUBLE, rank_next, 0, recv.data(), localSize[0] * localSize[1], MPI_DOUBLE, rank_prev, 0, commCart, MPI_STATUS_IGNORE);
    for(int j = 1; j <= localSize[1]; j++){
        for(int k = 1; k <= localSize[0]; k++){
            localSol[0 * sizeY * sizeX + j * sizeX + k] = recv[(j - 1) * localSize[0] + (k - 1)];
        }
    }
}

int main(int argc, char *argv[]){
    auto startTime = std::chrono::high_resolution_clock::now();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);    // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    // Get the rank of the process

    MPI_Dims_create(size, ndims, dims);                                     // Create a division of processes in a Cartesian grid
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &commCart);    // Create a Cartesian communicator
    MPI_Cart_coords(commCart, rank, ndims, coord);                          // Get the coordinates of the process in the communicator
    
    // Print auxiliary information
    if (rank == 0){
        std::cout << "============================================" << std::endl;
        std::cout << "Dimensions of the Cartesian grid: " << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;
    }

    // Divide domain points in subdomains
    int nxSubdomain = n / dims[0];       // Number of points in the x-direction of one of the 1st (dims[0]-1) subdomains
    int nySubdomain = n / dims[1];       // Number of points in the y-direction of one of the 1st (dims[1]-1) subdomains
    int nzSubdomain = n / dims[2];       // Number of points in the z-direction of one of the 1st (dims[2]-1) subdomains
    points = {nxSubdomain, nySubdomain, nzSubdomain};

    for(int d = 0; d < ndims; d++){
        start[d] = coord[d] * points[d];
        if(coord[d] != dims[d] - 1)
            end[d] = (coord[d] + 1) * points[d] - 1;
        else
            end[d] = n - 1;
    }

    localSize = {end[0] - start[0] + 1, end[1] - start[1] + 1, end[2] - start[2] + 1};

    // Print auxiliary information
    // std::cout << "Process: " << rank + 1 << "/" << size << " with coordinates: " 
    //           << coord[0] << " " << coord[1] << " " << coord[2] 
    //           << ", start = [" << start[0] << " " << start[1] << " " << start[2] << "], end = ["
    //           << end[0] << " " << end[1] << " " << end[2] << "]" << std::endl;

    double error = 0.0;
    double time;

    init();

    for(int t = 0; t < nt; t++){
        time = t * dt;
        rhs(time);
        xDirectionSolver(time);
        rotate(0);
        yDirectionSolver(time);
        rotate(1);
        zDirectionSolver(time);
        rotate(2);

        error = finalize(t);
        comunication();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    if(rank == 0){
        std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;  
        std::cout << "Error:        " << error << std::endl;  
        std::cout << "============================================" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
