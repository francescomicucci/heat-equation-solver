#include <cmath>
#include <chrono>
#include <iostream>

// Compute the error between the solution and the exact solution
double computeError(double* solution, double* exactSolution, const int dim){
    double error = 0.0;
    for(int i = 0; i < dim; i++){
        error += (solution[i] - exactSolution[i]) * (solution[i] - exactSolution[i]);
    }
    return std::sqrt(error) / dim;
}

int main(int argc, char* argv[]){
    const int nt = 6400;                        // Number of time steps
    const int nx = 40;                          // Number of grid points (start point and end point are not included)
    const double len = 1.0;                     // Length of the domain
    const double dt = 1.0 / nt;                 // Time step size
    const double dx = len / (nx + 1);           // Grid spacing
    const double coeff = dt / (dx * dx);    

    double time = 0.0;                          // Current time
    double* solution = new double[nx + 2];      // Solution of the linear system at the current time step
    double* oldSolution = new double[nx + 2];   // Solution of the linear system at the previous time step
    double* exactSolution = new double[nx + 2]; // Exact solution of the linear system at time = 1.0

    // Init rhs of the linear system and its exact solution
    for (int i = 0; i < nx + 2; i++){
        oldSolution[i] = 0.0;
        exactSolution[i] = std::sin(i * dx) * std::sin(1.0); 
    }

    auto start = std::chrono::high_resolution_clock::now();

    while(time < 1.0){
        // Update time
        time = time + dt;

        // Update solution
        solution[0] = 0.0;
        for(int i = 1; i < nx + 1; i++){
            solution[i] = oldSolution[i] * (1.0 - 2.0 * coeff) + coeff * (oldSolution[i - 1] + oldSolution[i + 1]) + 
                        dt * std::sin(i * dx) * (std::sin(time) + std::cos(time));
        }
        solution[nx + 1] = std::sin(1.0) * std::sin(time);

        // Update oldSolution
        for(int i = 0; i < nx + 2; i++){
            oldSolution[i] = solution[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> programTime = end - start;

    std::cout << "============================================" << std::endl;
    std::cout << "Error with " << nx << " points: " << computeError(solution, exactSolution, nx) << std::endl;
    std::cout << "Time: " << programTime.count() * 1000 << " milliseconds " << std::endl;
    std::cout << "Seconds / (nx * nt) = " << (programTime.count())/(nx * nt) << std::endl;
    std::cout << "============================================" << std::endl;

    // Free memory
    delete[] solution;
    delete[] oldSolution;
    delete[] exactSolution;

    return 0;
}