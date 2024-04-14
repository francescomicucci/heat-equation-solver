#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>

// Array containing the number of points for which we need to solve the heat eq. 
const int N[6] = {100, 1000, 10000, 100000, 1000000, 10000000};

// Implementation of the Thomas Algorithm.
// Used to solve a linear system of the form: A * solution = rhs
// where A is a tridiagonal matrix.
void thomasAlgorithm(   
                    double* solution,
                    double* diag,
                    double* upperDiag,
                    double* lowerDiag,
                    double* rhs,
                    int dim){
    double w;
    
    for(int i = 1; i < dim; i++){
        w = lowerDiag[i - 1] / diag[i - 1];
        diag[i] -= w * upperDiag[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    solution[dim - 1] = rhs[dim - 1] / diag[dim - 1];
    for(int i = dim - 2; i >= 0; i--){
        solution[i] = (rhs[i] - upperDiag[i] * solution[i + 1]) / diag[i];
    }
}

int main(int argc, char* argv[]){
    // Output stream to write data.csv
    std::ofstream fout("data.csv"); 
    fout << "n," << "deltaX," << "SecondsPerPoint," << "err" << "\n";

    for(int n:N){
        double dx = 1.0 / (n + 1);              // Step size
        double error = 0.0;                     // Error of the solution
        double* diag = new double[n];           // Main diagonal of the linear system
        double* upperDiag = new double[n - 1];  // 1st upper diagonal of the linear system
        double* lowerDiag = new double[n - 1];  // 1st lower diagonal of the linear system
        double* rhs = new double[n];            // Right-hand side of the linear system
        double* solution = new double[n];       // Calculated solution
        double* exactSolution = new double[n];  // Exact solution of the linear system

        auto start = std::chrono::high_resolution_clock::now();

        // Init matrix of the linear system
        std::fill_n(diag, n, 2.0 / (dx * dx));
        std::fill_n(upperDiag, n - 1, -1.0 / (dx * dx));
        std::fill_n(lowerDiag, n - 1, -1.0 / (dx * dx));

        // Init rhs of the linear system and its exact solution
        for (int i = 0; i < n; i++){
            if(i != n - 1)
                rhs[i] = std::sin((i + 1) * dx);
            else
                rhs[i] = std::sin(1.0)/(dx * dx) + std::sin((i + 1) * dx);          
            
            exactSolution[i] = std::sin((i + 1) * dx); 
        }

        // Use Thomas algorithm to solve the linear system
        thomasAlgorithm(solution, diag, upperDiag, lowerDiag, rhs, n);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time = end - start;

        // Compute error
        for(int i=0; i<n; i++)
            error += (solution[i] - exactSolution[i]) * (solution[i] - exactSolution[i]);
        error = std::sqrt(error);
        error = error / n;

        std::cout << "============================================" << std::endl;
        std::cout << "Error with " << n << " points: " << error << std::endl; 
        std::cout << "Time: " << time.count()*1000 << " milliseconds " << std::endl;
        std::cout << "Seconds per point: " << (time.count())/n << std::endl;
        std::cout << "============================================" << std::endl;

        fout << n << "," << dx << "," << (time.count())/n << "," << error << "\n";

        // Deallocate memory
        delete[] diag;
        delete[] lowerDiag;
        delete[] upperDiag;
        delete[] rhs;
        delete[] solution;
        delete[] exactSolution;
    }

    fout.close();
}