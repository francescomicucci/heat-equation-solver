#include "parallelHeat.hpp"

void ParallelHeat::solve()
{
    double time;

    for(int t = 0; t < nt; t++){
        time = t * dt;
        localRhsInit(time);
        xDirectionSolver(time);
        yDirectionSolver(time);
        zDirectionSolver(time);
        finalize();
    }
    computeError();
}

void ParallelHeat::localRhsInit(double &time)
{
    int index;

    for(int i = start[0]; i <= end[0]; i++){
        for(int j = start[1]; j <= end[1]; j++){
            for(int k = start[2]; k <= end[2]; k++){
                index = (i - start[0]) + (j - start[1]) * localSize[0] 
                      + (k - start[2]) * localSize[0] * localSize[1];
                
                localSol[index] = dt * (fFunction(i, j, k, time) + fFunction(i, j, k, time + dt)) / 2.0;
                localSol[index] -= 6.0 * timeStepSolution[getGlobalIndex(i, j, k)] * coeff;

                if(i > 0) localSol[index] += coeff * timeStepSolution[getGlobalIndex(i - 1, j, k)];
                if(i < nx - 1) localSol[index] += coeff * timeStepSolution[getGlobalIndex(i + 1, j, k)];
                else localSol[index] += coeff * sin(1.0) * sin((j + 1) * dx) * sin((k + 1) * dx) * sin(time);

                if(j > 0) localSol[index] += coeff * timeStepSolution[getGlobalIndex(i, j - 1, k)];
                if(j < nx - 1) localSol[index] += coeff * timeStepSolution[getGlobalIndex(i, j + 1, k)];
                else localSol[index] += coeff * sin((i + 1) * dx) * sin(1.0) * sin((k + 1) * dx) * sin(time);

                if(k > 0) localSol[index] += coeff * timeStepSolution[getGlobalIndex(i, j, k - 1)];
                if(k < nx - 1) localSol[index] += coeff * timeStepSolution[getGlobalIndex(i, j, k + 1)];
                else localSol[index] += coeff * sin((i + 1) * dx) * sin((j + 1) * dx) * sin(1.0) * sin(time);
            }
        }
    }
}

void ParallelHeat::xDirectionSolver(double &time)
{
    int index;                  // Local index of the point

    // BCs on x
    if(end[0] == nx - 1){
        for(int k = start[2]; k <= end[2]; k++){
            for(int j = start[1]; j <= end[1]; j++){
                index = (nx - 1 - start[0]) + (j - start[1]) * localSize[0] 
                      + (k - start[2]) * localSize[0] * localSize[1];
                localSol[index] -= noDiagElem * sin(1.0) * sin((j + 1) * dx) * sin((k + 1) * dx) * (sin(time + dt) - sin(time));
            }
        }
    }

    thomasX();
}

void ParallelHeat::thomasX()
{
    int index;                  // Local index of the point
    double prev, next;
    double diag[nx];            // b vector in the Wikipedia notation for the Thomas algorithm

    // Init the b vector in the Wikipedia notation for the Thomas algorithm
    diag[0] = diagElem;
    for(int i = 1; i < nx; i++){
        diag[i] = diagElem - noDiagElem * noDiagElem / diag[i - 1];
    }

    // Ranks of the source, previous and next processes
    int rank_source, rank_prev, rank_next; 

    MPI_Cart_shift(comm, 0, -1, &rank_source, &rank_prev);  // Get the rank of the previous process in the x-direction
    MPI_Cart_shift(comm, 0, 1, &rank_source, &rank_next);   // Get the rank of the next process in the x-direction

    // Forward sweep
    for(int k = start[2]; k <= end[2]; k++){
        for(int j = start[1]; j <= end[1]; j++){
            if(coord[0] != 0){
                MPI_Recv(&prev, 1, MPI_DOUBLE, rank_prev, getGlobalIndex(start[0], j, k), comm, MPI_STATUS_IGNORE);
            }

            for(int i = start[0]; i <= end[0]; i++){
                index = (i - start[0]) + (j - start[1]) * localSize[0] 
                      + (k - start[2]) * localSize[0] * localSize[1];
                
                if(i != 0) localSol[index] -= (noDiagElem * prev) / diag[i - 1];
                prev = localSol[index];
            }

            if(coord[0] != dims[0] - 1){
                MPI_Send(&prev, 1, MPI_DOUBLE, rank_next, getGlobalIndex(end[0] + 1, j, k), comm);
            }
        }
    }
    
    // Backward sweep
    for(int k = start[2]; k <= end[2]; k++){
        for(int j = start[1]; j <= end[1]; j++){
            if(coord[0] == dims[0] - 1){
                index = (end[0] - start[0]) + (j - start[1]) * localSize[0] 
                      + (k - start[2]) * localSize[0] * localSize[1];
                localSol[index] /= diag[localSize[0] - 1];
            }
            else{
                MPI_Recv(&next, 1, MPI_DOUBLE, rank_next, getGlobalIndex(end[0], j, k), comm, MPI_STATUS_IGNORE);
            }

            for(int i = end[0]; i >= start[0]; i--){
                index = (i - start[0]) + (j - start[1]) * localSize[0] 
                      + (k - start[2]) * localSize[0] * localSize[1];
                
                if(i != nx - 1) localSol[index] = (localSol[index] - noDiagElem * next) / diag[i];
                next = localSol[index];
            }

            if(coord[0] != 0){
                MPI_Send(&next, 1, MPI_DOUBLE, rank_prev, getGlobalIndex(start[0] - 1, j, k), comm);
            }
        }
    }
}

void ParallelHeat::rotateLocalSolFromXToY()
{
    int indexX;             // Local index of the point considering x as main direction
    int indexY;             // Local index of the point considering y as main direction

    double* temp = new double[localSize[0] * localSize[1] * localSize[2]];

    // Rotate from x to y
    for(int k = start[2]; k <= end[2]; k++){
        for(int j = start[1]; j <= end[1]; j++){
            for(int i = start[0]; i <= end[0]; i++){
                indexX = (i - start[0]) + (j - start[1]) * localSize[0] 
                       + (k - start[2]) * localSize[0] * localSize[1];
                indexY = (j - start[1]) + (k - start[2]) * localSize[1]
                       + (i - start[0]) * localSize[1] * localSize[2];
                temp[indexY] = localSol[indexX];
            }
        }
    }

    for(int i = 0; i < localSize[0] * localSize[1] * localSize[2]; i++){
        localSol[i] = temp[i];
    }

    delete[] temp;
}

void ParallelHeat::yDirectionSolver(double &time)
{
    int index;                  // Local index of the point

    // Rotate from x to y
    rotateLocalSolFromXToY();

    // BCs on y
    if(end[1] == nx - 1){
        for(int k = start[2]; k <= end[2]; k++){
            for(int i = start[0]; i <= end[0]; i++){
                index = (nx - 1 - start[1]) + (k - start[2]) * localSize[1]
                      + (i - start[0]) * localSize[1] * localSize[2];
                localSol[index] -= noDiagElem * sin((i + 1) * dx) * sin(1.0) * sin((k + 1) * dx) * (sin(time + dt) - sin(time));
            }
        }
    }

    thomasY();
}

void ParallelHeat::thomasY()
{
    int index;                  // Local index of the point
    double prev, next;
    double diag[nx];            // b vector in the Wikipedia notation for the Thomas algorithm

    // Init the b vector in the Wikipedia notation for the Thomas algorithm
    diag[0] = diagElem;
    for(int i = 1; i < nx; i++){
        diag[i] = diagElem - noDiagElem * noDiagElem / diag[i - 1];
    }

    // Ranks of the source, previous and next processes
    int rank_source, rank_prev, rank_next;

    MPI_Cart_shift(comm, 1, -1, &rank_source, &rank_prev);  // Get the rank of the previous process in the y-direction
    MPI_Cart_shift(comm, 1, 1, &rank_source, &rank_next);   // Get the rank of the next process in the y-direction

    // Forward sweep
    for(int i = start[0]; i <= end[0]; i++){
        for(int k = start[2]; k <= end[2]; k++){
            if(coord[1] != 0){
                MPI_Recv(&prev, 1, MPI_DOUBLE, rank_prev, getGlobalIndex(i, start[1], k), comm, MPI_STATUS_IGNORE);
            }

            for(int j = start[1]; j <= end[1]; j++){
                index = (j - start[1]) + (k - start[2]) * localSize[1]
                      + (i - start[0]) * localSize[1] * localSize[2];
                
                if(j != 0) localSol[index] -= (noDiagElem * prev) / diag[j - 1];
                prev = localSol[index];
            }

            if(coord[1] != dims[1] - 1){
                MPI_Send(&prev, 1, MPI_DOUBLE, rank_next, getGlobalIndex(i, end[1] + 1, k), comm);
            }
        }
    }

    // Backward sweep
    for(int i = start[0]; i <= end[0]; i++){
        for(int k = start[2]; k <= end[2]; k++){
            if(coord[1] == dims[1] - 1){
                index = (end[1] - start[1]) + (k - start[2]) * localSize[1]
                      + (i - start[0]) * localSize[1] * localSize[2];
                localSol[index] /= diag[localSize[1] - 1];
            }
            else{
                MPI_Recv(&next, 1, MPI_DOUBLE, rank_next, getGlobalIndex(i, end[1], k), comm, MPI_STATUS_IGNORE);
            }

            for(int j = end[1]; j >= start[1]; j--){
                index = (j - start[1]) + (k - start[2]) * localSize[1]
                      + (i - start[0]) * localSize[1] * localSize[2];
                
                if(j != nx - 1) localSol[index] = (localSol[index] - noDiagElem * next) / diag[j];
                next = localSol[index];
            }

            if(coord[1] != 0){
                MPI_Send(&next, 1, MPI_DOUBLE, rank_prev, getGlobalIndex(i, start[1] - 1, k), comm);
            }
        }
    }
}

void ParallelHeat::rotateLocalSolFromYToZ()
{
    int indexY;             // Local index of the point considering y as main direction
    int indexZ;             // Local index of the point considering z as main direction

    double* temp = new double[localSize[0] * localSize[1] * localSize[2]];

    // Rotate from y to z
    for(int k = start[2]; k <= end[2]; k++){
        for(int j = start[1]; j <= end[1]; j++){
            for(int i = start[0]; i <= end[0]; i++){
                indexY = (j - start[1]) + (k - start[2]) * localSize[1]
                       + (i - start[0]) * localSize[1] * localSize[2];
                indexZ = (k - start[2]) + (i - start[0]) * localSize[2]
                       + (j - start[1]) * localSize[2] * localSize[0];
                temp[indexZ] = localSol[indexY];
            }
        }
    }

    for(int i = 0; i < localSize[0] * localSize[1] * localSize[2]; i++){
        localSol[i] = temp[i];
    }

    delete[] temp;
}

void ParallelHeat::zDirectionSolver(double &time)
{
    int index;                  // Local index of the point

    // Rotate from y to z
    rotateLocalSolFromYToZ();

    // BCs on z
    if(end[2] == nx - 1){
        for(int j = start[1]; j <= end[1]; j++){
            for(int i = start[0]; i <= end[0]; i++){
                index = (nx - 1 - start[2]) + (i - start[0]) * localSize[2]
                      + (j - start[1]) * localSize[2] * localSize[0];
                localSol[index] -= noDiagElem * sin((i + 1) * dx) * sin((j + 1) * dx) * sin(1.0) * (sin(time + dt) - sin(time));
            }
        }
    }

    thomasZ();
}

void ParallelHeat::thomasZ()
{
    int index;                  // Local index of the point
    double prev, next;
    double diag[nx];            // b vector in the Wikipedia notation for the Thomas algorithm

    // Init the b vector in the Wikipedia notation for the Thomas algorithm
    diag[0] = diagElem;
    for(int i = 1; i < nx; i++){
        diag[i] = diagElem - noDiagElem * noDiagElem / diag[i - 1];
    }

    // Ranks of the source, previous and next processes
    int rank_source, rank_prev, rank_next;

    MPI_Cart_shift(comm, 2, -1, &rank_source, &rank_prev);  // Get the rank of the previous process in the z-direction
    MPI_Cart_shift(comm, 2, 1, &rank_source, &rank_next);   // Get the rank of the next process in the z-direction

    // Forward sweep
    for(int j = start[1]; j <= end[1]; j++){
        for(int i = start[0]; i <= end[0]; i++){
            if(coord[2] != 0){
                MPI_Recv(&prev, 1, MPI_DOUBLE, rank_prev, getGlobalIndex(i, j, start[2]), comm, MPI_STATUS_IGNORE);
            }

            for(int k = start[2]; k <= end[2]; k++){
                index = (k - start[2]) + (i - start[0]) * localSize[2]
                      + (j - start[1]) * localSize[2] * localSize[0];

                if(k != 0) localSol[index] -= (noDiagElem * prev) / diag[k - 1];
                prev = localSol[index];
            }

            if(coord[2] != dims[2] - 1){
                MPI_Send(&prev, 1, MPI_DOUBLE, rank_next, getGlobalIndex(i, j, end[2] + 1), comm);
            }
        }
    }

    // Backward sweep
    for(int j = start[1]; j <= end[1]; j++){
        for(int i = start[0]; i <= end[0]; i++){
            if(coord[2] == dims[2] - 1){
                index = (end[2] - start[2]) + (i - start[0]) * localSize[2]
                      + (j - start[1]) * localSize[2] * localSize[0];
                localSol[index] /= diag[localSize[2] - 1];
            }
            else{
                MPI_Recv(&next, 1, MPI_DOUBLE, rank_next, getGlobalIndex(i, j, end[2]), comm, MPI_STATUS_IGNORE);
            }

            for(int k = end[2]; k >= start[2]; k--){
                index = (k - start[2]) + (i - start[0]) * localSize[2]
                      + (j - start[1]) * localSize[2] * localSize[0];
                
                if(k != nx - 1) localSol[index] = (localSol[index] - noDiagElem * next) / diag[k];
                next = localSol[index];
            }

            if(coord[2] != 0){
                MPI_Send(&next, 1, MPI_DOUBLE, rank_prev, getGlobalIndex(i, j, start[2] - 1), comm);
            }
        }
    }
}

void ParallelHeat::rotateLocalSolFromZToX()
{
    int indexZ;             // Local index of the point considering z as main direction
    int indexX;             // Local index of the point considering x as main direction

    double* temp = new double[localSize[0] * localSize[1] * localSize[2]];

    // Rotate from z to x
    for(int k = start[2]; k <= end[2]; k++){
        for(int j = start[1]; j <= end[1]; j++){
            for(int i = start[0]; i <= end[0]; i++){
                indexZ = (k - start[2]) + (i - start[0]) * localSize[2]
                       + (j - start[1]) * localSize[2] * localSize[0];
                indexX = (i - start[0]) + (j - start[1]) * localSize[0] 
                       + (k - start[2]) * localSize[0] * localSize[1];
                temp[indexX] = localSol[indexZ];
            }
        }
    }

    for(int i = 0; i < localSize[0] * localSize[1] * localSize[2]; i++){
        localSol[i] = temp[i];
    }

    delete[] temp;
}

void ParallelHeat::finalize()
{
    int index;                              // Local index of the point
    double* temp;                           // Temporary array to store the local solution
    int* tempStart = new int[ndims];
    int* tempEnd = new int[ndims];
    int tempLenght;

    // Rotate from z to x
    rotateLocalSolFromZToX();

    for(int sender = 0; sender < size; sender++){
        if(sender == rank){
            for(int i = 0; i < ndims; i++){
                tempStart[i] = start[i];
                tempEnd[i] = end[i];
            }
        }

        MPI_Bcast(tempStart, ndims, MPI_INT, sender, comm);
        MPI_Bcast(tempEnd, ndims, MPI_INT, sender, comm);

        tempLenght = (tempEnd[0] - tempStart[0] + 1) 
                   * (tempEnd[1] - tempStart[1] + 1) 
                   * (tempEnd[2] - tempStart[2] + 1);
        
        temp = new double[tempLenght];
        
        if(sender == rank){
            for(int i = 0; i < tempLenght; i++){
                temp[i] = localSol[i];
            }
        }

        MPI_Bcast(temp, tempLenght, MPI_DOUBLE, sender, comm);

        for(int k = tempStart[2]; k <= tempEnd[2]; k++){
            for(int j = tempStart[1]; j <= tempEnd[1]; j++){
                for(int i = tempStart[0]; i <= tempEnd[0]; i++){
                    index = (i - tempStart[0]) + (j - tempStart[1]) * (tempEnd[0] - tempStart[0] + 1)
                          + (k - tempStart[2]) * (tempEnd[0] - tempStart[0] + 1) * (tempEnd[1] - tempStart[1] + 1);
                    timeStepSolution[getGlobalIndex(i, j, k)] += temp[index];
                }
            }
        }
    }
}

void ParallelHeat::computeError()
{
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < nx; j++){
            for(int k = 0; k < nx; k++){
                error += (timeStepSolution[getGlobalIndex(i, j, k)] - sin((i + 1) * dx) * sin((j + 1) * dx) * sin((k + 1) * dx) * sin(1.0)) 
                       * (timeStepSolution[getGlobalIndex(i, j, k)] - sin((i + 1) * dx) * sin((j + 1) * dx) * sin((k + 1) * dx) * sin(1.0));
            }
        }
    }
    error = std::sqrt(error) / N;
    
    if(rank == 0)
        std::cout << "Error: " << error << std::endl;
}