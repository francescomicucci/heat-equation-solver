#ifndef PARALLELHEAT_HPP
#define PARALLELHEAT_HPP

#include <cmath>
#include <iostream>
#include <mpi.h>

using std::sin;
using std::cos;

/*
* Class implementing a parallel solver for the heat equation in the 3D case.
*/
class ParallelHeat
{
    public:
    // Constructor.
    ParallelHeat(const int &rank_, const int &size_, const int* dims_, 
                const int* coord_, const int* start_, const int* end_, 
                const int* localSize_, MPI_Comm comm_, const int nx_, const int nt_)
        : rank(rank_)
        , size(size_)
        , dims(dims_)
        , coord(coord_)
        , start(start_)
        , end(end_)
        , localSize(localSize_)
        , comm(comm_)
        , nx(nx_)
        , nt(nt_)
    {
        timeStepSolution = new double[N];
        localSol = new double[localSize[0] * localSize[1] * localSize[2]];
        error = 0.0;
        
        std::fill_n(timeStepSolution, N, 0.0);
        std::fill_n(localSol, localSize[0] * localSize[1] * localSize[2], 0.0);
    }

    ~ParallelHeat()
    {
        delete[] timeStepSolution;
        delete[] localSol;
    }

    // Solve the heat equation in 3D.
    void solve();

    protected:
    // Process specific variables
    const int ndims = 3;        // Number of dimensions of the Cartesian grid
    const int rank;             // Rank of the process
    const int size;             // Total number of processes
    const int* dims;            // Array specifying the number of processes in each dimension
    const int* coord;           // Array specifying the Cartesian coordinates of the process
    const int* start;           // Array specifying the starting index of the local domain
    const int* end;             // Array specifying the ending index of the local domain
    const int* localSize;       // Array specifying the size of the local domain

    MPI_Comm comm;              // Communicator

    // Problem specific variables
    const int nx;                       // Number of points in a generic direction
    const int N = nx * nx * nx;         // Total number of points   
    const int nt;                       // Number of time steps
    const double dx = 1.0 / (nx + 1);   // Step size
    const double dt = 1.0 / nt;         // Time step size

    const double coeff = dt / (dx * dx);
    const double diagElem = 1.0 + coeff;        // Value of the main diagonal elements of the linear system
    const double noDiagElem = - coeff / 2.0;    // Value of the 1st upper and lower diagonal elements of the linear system

    double* timeStepSolution;       // Solution at the current time step
    double* localSol;               // Local solution of the linear system  
    double error;                   // Error of the solution

    // Returns the matrix global index associated with a certain mesh point
    int getGlobalIndex(int i, int j, int k){
        return i + j * nx + k * nx * nx;
    }

    // Force function
    double fFunction(int x, int y, int z, double t){
        return sin((x + 1) * dx) * sin((y + 1) * dx) * sin((z + 1) * dx) * (3.0 * sin(t) + cos(t));
    }

    // Init local rhs into localSol
    void localRhsInit(double &time);

    // Solve the linear system in the x-direction
    void xDirectionSolver(double &time);

    // Solve the linear system in the y-direction
    void yDirectionSolver(double &time);

    // Solve the linear system in the z-direction
    void zDirectionSolver(double &time);

    // Thomas algorithm in the x-direction
    void thomasX();

    // Thomas algorithm in the y-direction
    void thomasY();

    // Thomas algorithm in the z-direction
    void thomasZ();

    // Update solution at the current time step
    void finalize();

    // Rotate the local solution to better suit operation on the Y direction
    void rotateLocalSolFromXToY();

    // Rotate the local solution to better suit operation on the Z direction
    void rotateLocalSolFromYToZ();

    // Rotate the local solution to better suit operation on the X direction
    void rotateLocalSolFromZToX();

    // Compute the error
    void computeError();
};

#endif