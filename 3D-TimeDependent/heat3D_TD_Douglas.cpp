#include <cmath>
#include <chrono>
#include <iostream>

using std::sin;

const int n = 100;                      // Number of mesh points in each direction excluding the boundary points.
const int N = n * n * n;                // Total number of points excluding boundary points.
const int nt = 100;                     // Number of time steps.
const double dx = 1.0 / (n + 1);        // Mesh spacing.
const double dt = 1.0 / nt;             // Time step size.

const double coeff = dt / (dx * dx);
const double diagElem = 1.0 + coeff;        // Value of the main diagonal elements of the linear system
const double noDiagElem = - coeff / 2.0;    // Value of the 1st upper and lower diagonal elements of the linear system

// Returns the matrix index associated with a certain mesh point
int getIndex(int i, int j, int k){
    return i + (j - 1) * n + (k - 1) * n * n - 1;
}

// Forcing term
double fFunction(int x, int y, int z, double t, double dx){
    return sin(x * dx) * sin(y * dx) * sin(z * dx) * (3.0 * sin(t) + cos(t));
}

// Returns the value of the exact solution in a certain point at a certain time
double exactSolution(int x, int y, int z, double t, double dx){
    return sin(x * dx) * sin(y * dx) * sin(z * dx) * sin(t);
}

int main(int argc, char* argv[]){
    double* diag = new double[N];               // Main diagonal of the linear system
    double* rhs = new double[N];                // Right-hand side of the linear system
    double* solution = new double[N];           // Array containing the solution of all the intermediate linear systems exploited at each time step
    double* timeStepSolution = new double[N];   // Solution at a certain time step

    std::fill_n(solution, N, 0.0);
    std::fill_n(timeStepSolution, N, 0.0);

    double time = 0.0;      // Current time
    int index;              // Index of the current point
    
    auto start = std::chrono::high_resolution_clock::now();

    // Time loop
    for(int t = 0; t < nt; t++){
        time = t * dt;      // Update time

        // Init rhs for the Thomas algorithm (x direction)
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= n; j++){
                for(int k = 1; k <= n; k++){
                    index = getIndex(i, j, k);
                    rhs[index] = dt * (fFunction(i, j, k, time, dx) + fFunction(i, j, k, time + dt, dx)) / 2.0;
                    rhs[index] -= 6.0 * timeStepSolution[index] * coeff;

                    if(i > 1) {rhs[index] += coeff * timeStepSolution[getIndex(i - 1, j, k)];}
                    if(i < n) {rhs[index] += coeff * timeStepSolution[getIndex(i + 1, j, k)];}
                    else {rhs[index] += coeff * sin(1.0) * sin(j * dx) * sin(k * dx) * sin(time);}

                    if(j > 1) {rhs[index] += coeff * timeStepSolution[getIndex(i, j - 1, k)];}
                    if(j < n) {rhs[index] += coeff * timeStepSolution[getIndex(i, j + 1, k)];}
                    else {rhs[index] += coeff * sin(1.0) * sin(i * dx) * sin(k * dx) * sin(time);}

                    if(k > 1) {rhs[index] += coeff * timeStepSolution[getIndex(i, j, k - 1)];}
                    if(k < n) {rhs[index] += coeff * timeStepSolution[getIndex(i, j, k + 1)];}
                    else {rhs[index] += coeff * sin(1.0) * sin(i * dx) * sin(j * dx) * sin(time);}
                }
            }
        }

        // BCs on x
        for(int k = 1; k < n; k++){
            for(int j = 1; j < n; j++){
                rhs[getIndex(n, j, k)] -= noDiagElem * sin(1.0) * sin(j * dx) * sin(k * dx) * (sin(time + dt) - sin(time));
            }
        }

        // BCs on y
        for(int k = 1; k < n; k++){
            for(int j = 1; j < n; j++){
                rhs[getIndex(j, n, k)] -= noDiagElem * sin(1.0) * sin(j * dx) * sin(k * dx) * (sin(time + dt) - sin(time));
            }
        }

        // BCs on z
        for(int k = 1; k < n; k++){
            for(int j = 1; j < n; j++){
                rhs[getIndex(j, k, n)] -= noDiagElem * sin(1.0) * sin(j * dx) * sin(k * dx) * (sin(time + dt) - sin(time));
            }
        }

        std::fill_n(diag, N, diagElem);

        // X direction Thomas algorithm
        for(int k = 1; k <= n; k++){
            for(int j = 1; j <= n; j++){
                for(int i = 2; i <= n; i++){
                    double w = noDiagElem / diag[getIndex(i - 1, j, k)];
                    diag[getIndex(i, j, k)] -= w * noDiagElem;
                    rhs[getIndex(i, j, k)] -= w * rhs[getIndex(i - 1, j, k)];
                }
            }
        }
        for(int k = 1; k <= n; k++){
            for(int j = 1; j <= n; j++){
                solution[getIndex(n, j, k)] = rhs[getIndex(n, j, k)] / diag[getIndex(n, j, k)];
                for(int i = n - 1; i > 0; i--){
                    solution[getIndex(i, j, k)] = (rhs[getIndex(i, j, k)] - noDiagElem * solution[getIndex(i + 1, j, k)]) / diag[getIndex(i, j, k)];
                }
            }
        }

        // Init rhs for the Thomas algorithm (y direction)
        for(int i = 0; i < N; i++) rhs[i] = solution[i];

        std::fill_n(diag, N, diagElem);
        
        // Y direction Thomas algorithm
        for(int k = 1; k <= n; k++){
            for(int j = 2; j <= n; j++){
                for(int i = 1; i <= n; i++){
                    double w = noDiagElem / diag[getIndex(i, j - 1, k)];
                    diag[getIndex(i, j, k)] -= w * noDiagElem;
                    rhs[getIndex(i, j, k)] -= w * rhs[getIndex(i, j - 1, k)];
                }
            }
        }
        for(int k = 1; k <= n; k++){
            for(int j = n - 1; j > 0; j--){
                for(int i = 1; i <= n; i++){
                    solution[getIndex(i, n, k)] = rhs[getIndex(i, n, k)] / diag[getIndex(i, n, k)];
                    solution[getIndex(i, j, k)] = (rhs[getIndex(i, j, k)] - noDiagElem * solution[getIndex(i, j + 1, k)]) / diag[getIndex(i, j, k)];
                }
            }
        }

        // Init rhs for the Thomas algorithm (z direction)
        for(int i = 0; i < N; i++) rhs[i] = solution[i];

        std::fill_n(diag, N, diagElem);

        // Z direction Thomas algorithm
        for(int k = 2; k <= n; k++){
            for(int j = 1; j <= n; j++){
                for(int i = 1; i <= n; i++){
                    double w = noDiagElem / diag[getIndex(i, j, k - 1)];
                    diag[getIndex(i, j, k)] -= w * noDiagElem;
                    rhs[getIndex(i, j, k)] -= w * rhs[getIndex(i, j, k - 1)];
                }
            }
        }
        for(int k = n - 1; k > 0; k--){
            for(int j = 1; j <= n; j++){
                for(int i = 1; i <= n; i++){
                    solution[getIndex(i, j, n)] = rhs[getIndex(i, j, n)] / diag[getIndex(i, j, n)];
                    solution[getIndex(i, j, k)] = (rhs[getIndex(i, j, k)] - noDiagElem * solution[getIndex(i, j, k + 1)]) / diag[getIndex(i, j, k)];
                }
            }
        }

        // Update timeStepSolution
        for(int i = 0; i < N; i++) timeStepSolution[i] += solution[i];
    }

    // Compute error
    double error = 0.0;
    for(int k = 1; k <= n; k++){
        for(int j = 1; j <= n; j++){
            for(int i = 1; i <= n; i++){
                error = (timeStepSolution[getIndex(i, j, k)] - exactSolution(i, j, k, 1.0, dx)) 
                      * (timeStepSolution[getIndex(i, j, k)] - exactSolution(i, j, k, 1.0, dx));
            }
        }
    }
    error = error / N;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = end - start;
    
    std::cout << "============================================" << std::endl;
    std::cout << "Error with " << N << " points: " << error << std::endl; 
    std::cout << "Total time: " << totalTime.count() << " seconds" << std::endl;
    std::cout << "Seconds / (N * nt): " << (totalTime.count())/(N * nt) << std::endl;
    std::cout << "============================================" << std::endl;

    // Free memory
    delete[] diag;
    delete[] rhs;
    delete[] solution;
    delete[] timeStepSolution;

    return 0;
}