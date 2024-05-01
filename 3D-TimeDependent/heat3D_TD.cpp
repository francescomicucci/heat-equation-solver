#include <cmath>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using SpVec = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;
using Solver = Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper>;
using std::sin;
using std::cos;

// Returns the matrix index associated with a certain mesh point
int getIndex(int i, int j, int k, int n){
    return (i + (j - 1) * n + (k - 1) * n * n) - 1;
}

// Forcing function
double fFuntion(int i, int j, int k, double time, double dx){
    return sin(i * dx) * sin(j * dx) * sin(k * dx) * (3.0 * sin(time) + cos(time));
}

// Returns the exact solution at time t in a certain mesh point
double exactSolution(int i, int j, int k, double time, double dx){
    return sin(i * dx) * sin(j * dx) * sin(k * dx) * sin(time);
}

// Computes the error between the numerical and exact solution
double computeError(SpVec u, int n, int N, double time, double dx){
    double error = 0.0;
    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= n; j++){
            for(int k = 1; k <= n; k++){
                error += pow(u.coeffRef(getIndex(i, j, k, n)) - exactSolution(i, j, k, time, dx), 2);
            }
        }
    }
    return error = sqrt(error) / N;
}

// Initializes the linear system matrix
void buildMatrix(SpMat &A, int n, double dx, double dt){
    double diagElem = 1.0 / dt + 3.0 / (dx * dx);     // Value of the main diagonal elements of the linear system
    double noDiagElem = - 1.0 / (2.0 * dx * dx);      // Value of the 1st upper and lower diagonal elements of the linear system

    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= n; j++){
            for(int k = 1; k <= n; k++){
                A.coeffRef(getIndex(i, j, k, n), getIndex(i, j, k, n)) = diagElem;

                if(i > 1) A.coeffRef(getIndex(i, j, k, n), getIndex(i - 1, j, k, n)) = noDiagElem;
                if(j > 1) A.coeffRef(getIndex(i, j, k, n), getIndex(i, j - 1, k, n)) = noDiagElem;
                if(k > 1) A.coeffRef(getIndex(i, j, k, n), getIndex(i, j, k - 1, n)) = noDiagElem;

                if(i <= n - 1) A.coeffRef(getIndex(i, j, k, n), getIndex(i + 1, j, k, n)) = noDiagElem;
                if(j <= n - 1) A.coeffRef(getIndex(i, j, k, n), getIndex(i, j + 1, k, n)) = noDiagElem;
                if(k <= n - 1) A.coeffRef(getIndex(i, j, k, n), getIndex(i, j, k + 1, n)) = noDiagElem;
            }
        }
    }
}

// Initializes the right hand side vector
void buildRhs(SpVec &b, SpVec &oldSolution, int n, double dx, double dt, double time){
    int index;
    for(int i = 0; i < n * n * n; i++){
        b.coeffRef(i) = 0.0;
    }
    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= n; j++){
            for(int k = 1; k <= n; k++){
                index = getIndex(i, j, k, n);

                b.coeffRef(index) += 0.5 * (fFuntion(i, j, k, time - dt, dx) + fFuntion(i, j, k, time, dx));
                b.coeffRef(index) += oldSolution.coeffRef(index) * (1.0 / dt - 3.0 / (dx * dx));

                if(i > 1) b.coeffRef(index) += 0.5 * (1.0 / (dx * dx)) * oldSolution.coeffRef(getIndex(i - 1, j, k, n));
                if(j > 1) b.coeffRef(index) += 0.5 * (1.0 / (dx * dx)) * oldSolution.coeffRef(getIndex(i, j - 1, k, n));
                if(k > 1) b.coeffRef(index) += 0.5 * (1.0 / (dx * dx)) * oldSolution.coeffRef(getIndex(i, j, k - 1, n));

                if(i < n) b.coeffRef(index) += 0.5 * (1.0 / (dx * dx)) * oldSolution.coeffRef(getIndex(i + 1, j, k, n));
                if(j < n) b.coeffRef(index) += 0.5 * (1.0 / (dx * dx)) * oldSolution.coeffRef(getIndex(i, j + 1, k, n));
                if(k < n) b.coeffRef(index) += 0.5 * (1.0 / (dx * dx)) * oldSolution.coeffRef(getIndex(i, j, k + 1, n));

                // BCs
                if(i == n) b.coeffRef(index) += 0.5 * (1.0 / (dx * dx)) * sin(1.0) * sin(j * dx) * sin(k * dx) * (sin(time - dt) + sin(time));
                if(j == n) b.coeffRef(index) += 0.5 * (1.0 / (dx * dx)) * sin(i * dx) * sin(1.0) * sin(k * dx) * (sin(time - dt) + sin(time));
                if(k == n) b.coeffRef(index) += 0.5 * (1.0 / (dx * dx)) * sin(i * dx) * sin(j * dx) * sin(1.0) * (sin(time - dt) + sin(time));
            }
        }
    }
}

// Solves the linear system
void solveSystem(SpMat &A, SpVec &b, SpVec &u){
    Solver solver;
    solver.compute(A);
    u = solver.solve(b);
}

int main(int argc, char* argv[]){
    const int nt = 100;
    const int n_[3] = {3, 10, 25};

    std::vector<double> error_;
    std::vector<double> dx_;

    for(int n:n_){
        int N = n * n * n;
        double dx = 1.0 / (n + 1);
        double dt = 1.0 / nt;
        double t = 0.0 + dt;
        double error;

        SpMat A(N, N);
        SpVec b(N);
        SpVec u(N);

        auto startBuildMatrix = std::chrono::high_resolution_clock::now();
        buildMatrix(A, n, dx, dt);
        auto endBuildMatrix = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> timeBuildMatrix = endBuildMatrix - startBuildMatrix;

        auto startSolve = std::chrono::high_resolution_clock::now();
        while(t <= 1.0){
            buildRhs(b, u, n, dx, dt, t);
            solveSystem(A, b, u);
            t += dt;
        }
        auto endSolve = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> timeSolve = endSolve - startSolve;

        error = computeError(u, n, N, 1.0, dx);

        error_.push_back(error);
        dx_.push_back(dx);

        std::cout << "===============================================================" << std::endl;
        std::cout << "Time to build the matrix: " << timeBuildMatrix.count() * 1000 << " milliseconds " << std::endl;
        std::cout << "Time to solve the problem: " << timeSolve.count() << " seconds " << std::endl;
        std::cout << "Error with " << N << " points: " << error << std::endl;
    }

    std::cout << "===============================================================" << std::endl;
    std::cout << "Convergence rate: " << std::endl;
    for(int i=0; i < 2; i++){
        std::cout << (log(error_.at(i + 1)) - log(error_.at(i)))/(log(dx_.at(i + 1)) - log(dx_.at(i))) << std::endl;
    }
    std::cout << "===============================================================" << std::endl;   
}