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
#if defined(__RK4)
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Nx, RK_data_struct* RK_data);
#endif
#if defined(__RK5) || defined(__DPRK5)
void RK5DPStep(const double dt, const long int* N, const int iters, const ptrdiff_t local_Nx, RK_data_struct* RK_data);
#endif
#if defined(__DPRK5)
double DPMax(double a, double b);
double DPMin(double a, double b);
#endif
void NonlinearRHS(fftw_complex* psi_hat, fftw_complex* nonlin_term, fftw_complex* grad_psi_hat, double* grad_psi);
void ExactSoln(double t);
void PrintUpdateToTerminal(int iters, double t, double dt, double T, int save_data_indx);
void ComputeSystemMeasurables(double t, int iter, RK_data_struct* RK_data);
void InitializeSystemMeasurables(RK_data_struct* RK_data);
void SystemCheck(double dt, int iters);
void GetTimestep(double* dt);
double GetMaxData(char* dtype);
void InitializeSpaceVariables(double** x, int** k, const long int* N);
void InitialConditions(double* psi, fftw_complex* psi_hat, const long int* N);
void InitializeIntegrationVariables(double* t0, double* t, double* dt, double* T, long int* trans_steps);
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N);
void AllocateMemory(const long int* NBatch, RK_data_struct* RK_data);
void InitializeFFTWPlans(const long int* N);
void FreeMemory(RK_data_struct* RK_data);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------