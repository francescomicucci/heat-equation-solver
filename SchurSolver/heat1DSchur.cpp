#include "heat1DSchur.hpp"

void SchurSolver::solve()
{
    setup();
    computeSchurComplement();
    updateRhsInterface();
    computeSolution();
    computeError();
}

void SchurSolver::setup()
{
    rhsInterface  = new double[numDecomp - 1];       // Allocate memory for rhsInterface
    solution      = new double[n];                   // Allocate memory for solution
    yStorage      = new double[n - numDecomp + 1];   // Allocate memory for yStorage
    diagS         = new double[numDecomp - 1];       // Allocate memory for diagS
    upperDiagS    = new double[numDecomp - 2];       // Allocate memory for upperDiagS
    lowerDiagS    = new double[numDecomp - 2];       // Allocate memory for lowerDiagS

    xi = new double[dimSubmatrix];                   // Allocate memory for xi
    yi = new double[dimSubmatrix];                   // Allocate memory for yi
    xm = new double[dimLatestSubmatrix];             // Allocate memory for xm
    ym = new double[dimLatestSubmatrix];             // Allocate memory for ym

    extractRhsInterface();                           // Extract rhsInterface from rhs
}

void SchurSolver::computeSchurComplement()
{
    schurSubsystemsSolver(xi, yi, diagElem, upperElem, lowerElem, dimSubmatrix);
    schurSubsystemsSolver(xm, ym, diagElem, upperElem, lowerElem, dimLatestSubmatrix);

    // When we perform the following matrix multiplication Ei*Ai^(-1)*Di we get as result the following matrix:
    //                        i-1   i
    //                  |                   |
    // Ei*Ai^(-1)*Di =  |     s11  s12      |    i-1
    //                  |     s21  s22      |    i
    //                  |                   |  
    double s11 = upperElem * xi[0];
    double s12 = upperElem * yi[0];
    double s21 = lowerElem * xi[dimSubmatrix - 1];
    double s22 = lowerElem * yi[dimSubmatrix - 1];

    // When we perform the following matrix multiplication En-1*An-1^(-1)*Dn-1 we get as result the following matrix:
    //                                        n-2            
    //                        |                   |
    // En-1*An-1^(-1)*Dn-1 =  |                   |    
    //                        |                   |    
    //                        |               smm |    n-2
    double smm = upperElem * xm[0];

    // Assembling Schur complement matrix: S = F - E0*A0^(-1)*D0 - E1*A1^(-1)*D1 - ...
    // Obs: F = diag(diagElem)
    std::fill_n(diagS, numDecomp, diagElem);
    diagS[0] -= s22;
    diagS[numDecomp - 2] -= smm;
    for(int i = 1; i < numDecomp - 1; i++){
        diagS[i-1]      -= s11;
        diagS[i]        -= s22;
        upperDiagS[i-1] -= s12;
        lowerDiagS[i-1] -= s21;
    }
}

void SchurSolver::updateRhsInterface()
{
    double* subrhs = new double[dimSubmatrix];               // Rhs elements associated with the Matrix Ai
    double* y = new double[dimSubmatrix];                    // Solution of the linear system: Ai * y = subrhs 
    double* diag = new double[dimSubmatrix];                 // Main diagonal of the Matrix Ai
    double* upperDiag = new double[dimSubmatrix - 1];        // 1st upper diagonal of the Matrix Ai
    double* lowerDiag = new double[dimSubmatrix - 1];        // 1st lower diagonal of the Matrix Ai
    
    // Init diag, upperDiag, lowerDiag, subrhs of Matrix A0
    std::fill_n(diag, dimSubmatrix, diagElem);
    std::fill_n(upperDiag, dimSubmatrix - 1, upperElem);
    std::fill_n(lowerDiag, dimSubmatrix - 1, lowerElem);
    extractRhsAj(subrhs, 0);

    // Solve A0 * y = subrhs 
    thomasAlgorithm(y, diag, upperDiag, lowerDiag, subrhs, dimSubmatrix);

    // Store y
    for(int j = 0; j < dimSubmatrix; j++) yStorage[j] = y[j];
    
    // Update rhsInterface
    rhsInterface[0] -= lowerElem * y[dimSubmatrix - 1]; 

    for(int i = 1; i < numDecomp - 1; i++){
        // Init diag, upperDiag, lowerDiag, subrhs of Matrix Ai
        std::fill_n(diag, dimSubmatrix, diagElem);
        std::fill_n(upperDiag, dimSubmatrix - 1, upperElem);
        std::fill_n(lowerDiag, dimSubmatrix - 1, lowerElem); 
        extractRhsAj(subrhs, i);
        
        // Solve Ai * y = subrhs 
        thomasAlgorithm(y, diag, upperDiag, lowerDiag, subrhs, dimSubmatrix);

        // Store y
        for(int j = 0; j < dimSubmatrix; j++) yStorage[i * dimSubmatrix + j] = y[j];

        // Update rhsInterface
        rhsInterface[i - 1] -= upperElem * y[0];
        rhsInterface[i]     -= lowerElem * y[dimSubmatrix - 1]; 
    }

    subrhs = new double[dimLatestSubmatrix];
    y = new double[dimLatestSubmatrix];
    diag = new double[dimLatestSubmatrix];
    upperDiag = new double[dimLatestSubmatrix - 1];
    lowerDiag = new double[dimLatestSubmatrix - 1];

    // Init diag, upperDiag, lowerDiag, subrhs of Matrix An-1
    std::fill_n(diag, dimLatestSubmatrix, diagElem);
    std::fill_n(upperDiag, dimLatestSubmatrix - 1, upperElem);
    std::fill_n(lowerDiag, dimLatestSubmatrix - 1, lowerElem);
    extractRhsAj(subrhs, numDecomp - 1);

    // Solve An-1 * y = subrhs 
    thomasAlgorithm(y, diag, upperDiag, lowerDiag, subrhs, dimLatestSubmatrix);

    // Store y
    for(int j = 0; j < dimLatestSubmatrix; j++) yStorage[(numDecomp - 1) * dimSubmatrix + j] = y[j];

    // Update rhsInterface
    rhsInterface[numDecomp - 2] -= upperElem * y[0];

    // Free memory
    delete[] subrhs;
    delete[] y;
    delete[] diag;
    delete[] upperDiag;
    delete[] lowerDiag;
}

void SchurSolver::computeSolution()
{
    double* xInt = new double[numDecomp - 1];           // Solution of the linear system S * xInt = rhsInterface       

    // Solve S * xInt = rhsInterface
    thomasAlgorithm(xInt, diagS, upperDiagS, lowerDiagS, rhsInterface, numDecomp - 1);

    // Compute the solution for all the points
    for(int j = 0; j < numDecomp - 1; j++) solution[(j + 1) * dimSubmatrix + j] = xInt[j];
    for(int j = 0; j < dimSubmatrix; j++){
        solution[j] = yStorage[j];
        solution[j] -= yi[j] * xInt[0];
    }
    for(int i = 1; i < numDecomp - 1; i++){
        for(int j = 0; j < dimSubmatrix; j++){
            solution[i * (dimSubmatrix + 1) + j] = yStorage[i * dimSubmatrix + j];
            solution[i * (dimSubmatrix + 1) + j] -= xi[j] * xInt[i - 1];
            solution[i * (dimSubmatrix + 1) + j] -= yi[j] * xInt[i];
        }     
    }
    for(int j = 0; j < dimLatestSubmatrix; j++){
        solution[(numDecomp - 1) * (dimSubmatrix + 1) + j] = yStorage[(numDecomp - 1) * dimSubmatrix + j];
        solution[(numDecomp - 1) * (dimSubmatrix + 1) + j] -= xm[j] * xInt[numDecomp - 2];
    }  

    // Free memory
    delete[] xInt;
}

void SchurSolver::computeError()
{
    for(int i = 0; i < n; i++)
        error += (solution[i] - exactSolution[i]) * (solution[i] - exactSolution[i]);
    error = std::sqrt(error);
    error = error / n;
}

void SchurSolver::thomasAlgorithm(double* solution, double* diag,
                    double* upperDiag, double* lowerDiag,
                    double* rhs, int dim)
{
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

void SchurSolver::schurSubsystemsSolver(double x[], double y[], 
                        double diag, double upperDiag, 
                        double lowerDiag, int dim)
{
    // Wikipedia notation
    double w;
    double* b = new double[dim];
    double* d = new double[dim];

    std::fill_n(b, dim, diag);
    d[0] = lowerDiag;

    for(int i = 1; i < dim; i++){
        w = lowerDiag / b[i - 1];
        b[i] = b[i] - w * upperDiag;
        d[i] = d[i] - w * d[i-1];
    }
    x[dim - 1] = d[dim - 1] / b[dim - 1];
    y[dim - 1] = upperDiag / b[dim - 1];
    for(int i = dim - 2; i >= 0; i--){
        x[i] = (d[i] - upperDiag * x[i + 1]) / b[i];
        y[i] = (- upperDiag * y[i + 1]) / b[i];
    }
}

void SchurSolver::extractRhsInterface(){
    int stride = (n - numDecomp + 1) / numDecomp;
    
    for(int i = 1; i < numDecomp; i++){
        rhsInterface[i - 1] = rhs[i * (stride + 1) - 1];
    }
}

void SchurSolver::extractRhsAj(double rhsAi[], int j){
    int stride = (n - numDecomp + 1) / numDecomp, lenghtRhsAi = stride;
    if(j == numDecomp - 1) lenghtRhsAi = n - numDecomp + 1 - ((numDecomp - 1) * stride);
    
    for(int i = 0; i < lenghtRhsAi; i++){
        rhsAi[i] = rhs[i + j * (stride + 1)];
    }
}