/**
* @file solver.h
* @author Enda Carroll
* @date Mar 2022
* @brief Header file containing the function prototypes for the solver.c file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "data_types.h"




// ---------------------------------------------------------------------
//  Function Prototpyes
// ---------------------------------------------------------------------
// Main function for the pseudospectral solver
void SpectralSolve(void);
void InitializeSpaceVariables(double** x, int** k, const long int* N);
void InitialConditions(double* u, fftw_complex* u_hat, const long int* N);
void InitializeIntegrationVariables(double* t0, double* t, double* dt, double* T, long int* trans_steps);
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N);
void AllocateMemory(const long int* NBatch, RK_data_struct* RK_data);
void InitializeFFTWPlans(const long int* N);
void FreeMemory(RK_data_struct* RK_data);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------