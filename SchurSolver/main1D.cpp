#include <chrono>
#include <iostream>
#include "heat1DSchur.hpp"

// Initialize rhs and exact solution of the pb.
void initialize(double rhs[], double exactSolution[], const int &n, const double &dx)
{
    for (int i = 0; i < n - 1; i++){
        exactSolution[i] = std::sin((i + 1) * dx); 
        rhs[i] = std::sin((i + 1) * dx);
    }
    exactSolution[n-1] = std::sin(n * dx);
    rhs[n-1] = std::sin(1.0) / (dx * dx) + std::sin(n * dx); 
}

// Main function
int main(int argc, char *argv[])
{   
    const int n           = 1000000;        // Number of points
    const int numDecomp   = 5;              // Number of decomposition of the system
    double dx             = 1.0 / (n + 1);  // Step size
    double* rhs           = new double[n];  // Rhs of the problem
    double* exactSolution = new double[n];  // Exact solution of the problem
    double diagElem = 2.0 / (dx * dx);      // Diagonal element of the matrix
    double upperElem = -1.0 / (dx * dx);    // Upper diagonal element of the matrix
    double lowerElem = -1.0 / (dx * dx);    // Lower diagonal element of the matrix

    auto start = std::chrono::high_resolution_clock::now();
    
    initialize(rhs, exactSolution, n, dx);
    SchurSolver solver(n, numDecomp, dx, rhs, exactSolution, diagElem, upperElem, lowerElem);
    solver.solve();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;

    std::cout << "============================================" << std::endl;
    std::cout << "Error with " << n << " points: " << solver.getError() << std::endl;
    std::cout << "Time: " << time.count() * 1000 << " milliseconds " << std::endl;
    std::cout << "Seconds per point: " << time.count() / n << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}