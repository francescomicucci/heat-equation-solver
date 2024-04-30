#ifndef SCHURSOLVER_HPP
#define SCHURSOLVER_HPP

#include <cmath>
#include <iostream>

/*
* Class implementing the Schur complement method for solving the heat equation.
*/
class SchurSolver
{
    public:
    // Constructor.
    SchurSolver(const int &n_, const int &numDecomp_,
                const double &dx_, const double* rhs_, 
                const double* exactSolution_, double diagElem_,
                double upperElem_, double lowerElem_)
        : n(n_)
        , numDecomp(numDecomp_)
        , dx(dx_)
        , rhs(rhs_)
        , exactSolution(exactSolution_)
        , diagElem(diagElem_)
        , upperElem(upperElem_)
        , lowerElem(lowerElem_)
    {
        error = 0.0;
        dimSubmatrix = (n - numDecomp + 1) / numDecomp; 
        dimLatestSubmatrix = n - numDecomp + 1 - ((numDecomp - 1) * dimSubmatrix);
    }

    // Solve the heat equation in 1D.
    void solve();

    // Return the error associated with the computed solution.
    double getError() const { return error; }

    protected:
    const int n;                    // Number of points
    const int numDecomp;            // Number of decomposition of the system
    const double dx;                // Step size
    const double* rhs;              // Rhs of the problem
    const double* exactSolution;    // Exact solution of the problem
    double diagElem;                // Value of the diagonal elements of the initial matrix
    double upperElem;               // Value of the elements of the 1st upper diagonal of the initial matrix
    double lowerElem;               // Value of the elements of the 1st lower diagonal of the initial matrix
    double error;                   // Error of the computed solution
    double* solution;               // Solution of the problem
    double* rhsInterface;           // Rhs elements associated with the matrix S
    double* diagS;                  // Schur complement main diagonal
    double* upperDiagS;             // Schur complement 1st upper diagonal
    double* lowerDiagS;             // Schur complement 1st lower diagonal   
    int dimSubmatrix;               // Dim of the first n-1 submatrices A0, A1, ..., An-2
    int dimLatestSubmatrix;         // Dim of the latest submatrix An-1

    double* xi;                 // xi corresponds to the column i-1 of (Ai^(-1)*Di)
    double* yi;                 // yi corresponds to the column i of (Ai^(-1)*Di) 
    double* xm;                 // xm corresponds to the column n-2 of (An-1^(-1)*Dn-1)
    double* ym;                 // [in this case ym does not exist, we allocate memory just to use the function]


    // Vector storing the solutions of the linear systems: Ai * y = subrhs
    // The 1st element of the i-th solution is contained at the cell (i-1) * dimSubmatrix
    // The successive elements comes after the cell containing the 1st one.
    double* yStorage;

    // Initialize all the class components
    void setup();

    // Compute the elements componing the Schur complement
    void computeSchurComplement();

    // Update rhsInterface in such a way to take into account the Schur Complement computation
    void updateRhsInterface();

    // Compute the solution of the heat equation in 1D.
    void computeSolution();

    // Compute the error of the solution w.r.t. the exact solution.
    void computeError();

    // Implementation of the Thomas Algorithm.
    // Used to solve a linear system of the form: A * solution = rhs
    // where A is a tridiagonal matrix.
    void thomasAlgorithm(double* solution, double* diag,
                        double* upperDiag, double* lowerDiag,
                        double* rhs, int dim);

    // Modified version of the Thomas algorithm used for solving simultaneously 
    // 2 different linear systems having the same matrix A and rhs as follow:
    //      |b c        |
    //      |a b c      |
    //  A = |  a b c    |
    //      |    a b c  |
    //      |      a b c|
    // The rhs of the 2 systems are the following:
    //          |a|             |0|
    //          |0|             |0|
    //  rhs1 =  |0|      rhs2 = |0|
    //          |0|             |0|
    //          |0|             |c|
    void schurSubsystemsSolver(double x[], double y[], 
                            double diag, double upperDiag, 
                            double lowerDiag, int dim);

    // Extract the rhs elements associated with the matrix S from the rhs of the problem.
    void extractRhsInterface();

    // Extract the rhs elements associated with the matrix Ai from the rhs of the problem. 
    void extractRhsAj(double rhsAi[], int j);
};

#endif