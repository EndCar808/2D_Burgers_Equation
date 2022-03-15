/**
* @file solver.c 
* @author Enda Carroll
* @date Jun 2022
* @brief file containing the main functions used in the pseudopectral method
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_funcs.h"
#include "utils.h"
#include "solver.h"

// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
// Define RK4 variables
#if defined(__RK4)
static const double RK4_C2 = 0.5, 	  RK4_A21 = 0.5, \
				  	RK4_C3 = 0.5,	           					RK4_A32 = 0.5, \
				  	RK4_C4 = 1.0,                      									   RK4_A43 = 1.0, \
				              	 	  RK4_B1 = 1.0/6.0, 		RK4_B2  = 1.0/3.0, 		   RK4_B3  = 1.0/3.0, 		RK4_B4 = 1.0/6.0;
// Define RK5 Dormand Prince variables
#elif defined(__RK5) || defined(__DPRK5)
static const double RK5_C2 = 0.2, 	  RK5_A21 = 0.2, \
				  	RK5_C3 = 0.3,     RK5_A31 = 3.0/40.0,       RK5_A32 = 0.5, \
				  	RK5_C4 = 0.8,     RK5_A41 = 44.0/45.0,      RK5_A42 = -56.0/15.0,	   RK5_A43 = 32.0/9.0, \
				  	RK5_C5 = 8.0/9.0, RK5_A51 = 19372.0/6561.0, RK5_A52 = -25360.0/2187.0, RK5_A53 = 64448.0/6561.0, RK5_A54 = -212.0/729.0, \
				  	RK5_C6 = 1.0,     RK5_A61 = 9017.0/3168.0,  RK5_A62 = -355.0/33.0,     RK5_A63 = 46732.0/5247.0, RK5_A64 = 49.0/176.0,    RK5_A65 = -5103.0/18656.0, \
				  	RK5_C7 = 1.0,     RK5_A71 = 35.0/384.0,								   RK5_A73 = 500.0/1113.0,   RK5_A74 = 125.0/192.0,   RK5_A75 = -2187.0/6784.0,    RK5_A76 = 11.0/84.0, \
				              		  RK5_B1  = 35.0/384.0, 							   RK5_B3  = 500.0/1113.0,   RK5_B4  = 125.0/192.0,   RK5_B5  = -2187.0/6784.0,    RK5_B6  = 11.0/84.0, \
				              		  RK5_Bs1 = 5179.0/57600.0, 						   RK5_Bs3 = 7571.0/16695.0, RK5_Bs4 = 393.0/640.0,   RK5_Bs5 = -92097.0/339200.0, RK5_Bs6 = 187.0/2100.0, RK5_Bs7 = 1.0/40.0;
#endif
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Main function that performs the pseudospectral solver
 */
void SpectralSolve(void) {

	// Initialize variables
	const long int N[SYS_DIM]      = {sys_vars->N[0], sys_vars->N[1]};
	const long int NBatch[SYS_DIM] = {sys_vars->N[0], sys_vars->N[1] / 2 + 1};
	double norm[2];

	// Initialize the Runge-Kutta struct
	struct RK_data_struct* RK_data;	   // Initialize pointer to a RK_data_struct
	struct RK_data_struct RK_data_tmp; // Initialize a RK_data_struct
	RK_data = &RK_data_tmp;		       // Point the ptr to this new RK_data_struct

	// -------------------------------
	// Allocate memory
	// -------------------------------
	AllocateMemory(NBatch, RK_data);

	// -------------------------------
	// FFTW Plans Setup
	// -------------------------------
	InitializeFFTWPlans(N);

	// -------------------------------
	// Initialize the System
	// -------------------------------
	// Initialize the collocation points and wavenumber space 
	InitializeSpaceVariables(run_data->x, run_data->k, N);

	// Get initial conditions
	InitialConditions(run_data->psi, run_data->psi_hat, N, norm);

	// -------------------------------
	// Integration Variables
	// -------------------------------
	// Initialize integration variables
	double t0;
	double t;
	double dt;
	double T;
	long int trans_steps;
	#if defined(__DPRK5)
	int try   = 1;
	int tries = 0;
	double dt_new;
	#endif

	// Get timestep and other integration variables
	InitializeIntegrationVariables(&t0, &t, &dt, &T, &trans_steps);

	// -------------------------------
	// Create & Open Output File
	// -------------------------------
	// Inialize system measurables
	InitializeSystemMeasurables(RK_data);
	   
	// Create and open the output file - also write initial conditions to file
	CreateOutputFilesWriteICs(N, dt);

	// -------------------------------------------------
	// Print IC to Screen 
	// -------------------------------------------------
	#if defined(__PRINT_SCREEN)
	PrintUpdateToTerminal(0, t0, dt, T, 0, norm);
	#endif	
	
	//////////////////////////////
	// Begin Integration
	//////////////////////////////
	t         += dt;
	int iters = 1;
	#if defined(TRANSIENTS)
	int save_data_indx = 0;
	#else
	int save_data_indx = 1;
	#endif
	while (t <= T) {

		// -------------------------------	
		// Integration Step
		// -------------------------------
		#if defined(__RK4)
		RK4Step(dt, N, sys_vars->local_Nx, RK_data);
		#elif defined(__RK5)
		RK5DPStep(dt, N, iters, sys_vars->local_Nx, RK_data);
		#elif defined(__DPRK5)
		while (try) {
			if (tries < DP_MAX_TRIES) {
				// Try a Dormand Prince step and compute the local error
				RK5DPStep(dt, N, iters, sys_vars->local_Nx, RK_data);

				// Compute the new timestep
				dt_new = dt * DPMin(DP_DELTA_MAX, DPMax(DP_DELTA_MIN, DP_DELTA * pow(1.0 / RK_data->DP_err, 0.2)));
				
				// If error is bad repeat else move on
				if (RK_data->DP_err <= 1.0) {
					RK_data->DP_fails++;
					dt = dt_new;
					tries++;
					continue;
				}
				else {
					dt = dt_new;
					tries = 0;
					break;
				}
			}
			else {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Dorman Prince Solver reached its maximum no. of tries - Max Tries: ["CYAN"%d"RESET"] at t: ["CYAN"%lf"RESET"] iter: ["CYAN"%d"RESET"] with dt: ["CYAN"%lf"RESET"]\n-->> Exiting!!!\n", DP_MAX_TRIES, t, iters, dt);
				exit(1);
			}			
		}
		#endif
		
		// -------------------------------
		// Write To File
		// -------------------------------
		if ((iters > trans_steps) && (iters % sys_vars->SAVE_EVERY == 0)) {
			#if defined(TESTING)
			ExactSoln(t, norm);
			#endif

			// Record System Measurables
			ComputeSystemMeasurables(t, save_data_indx, RK_data);

			// Write the appropriate datasets to file
			WriteDataToFile(t, dt, save_data_indx);
			
			// Update saving data index
			save_data_indx++;
		}
		// -------------------------------
		// Print Update To Screen
		// -------------------------------
		#if defined(__PRINT_SCREEN)
		#if defined(TRANSIENTS)
		if (iters == trans_steps && !(sys_vars->rank)) {
			printf("\n\n...Transient Iterations Complete!\n\n");
		}
		#endif
		if (iters % sys_vars->SAVE_EVERY == 0) {
			#if defined(TRANSIENTS)
			if (iters <= sys_vars->trans_iters) {
				// If currently performing transient iters, call system measure for printing to screen
				ComputeSystemMeasurables(t, save_data_indx, RK_data);
			}
			#endif
			PrintUpdateToTerminal(iters, t, dt, T, save_data_indx - 1, norm);
		}
		#endif

		// -------------------------------
		// Update & System Check
		// -------------------------------
		// Update timestep & iteration counter
		iters++;
		#if defined(__ADAPTIVE_STEP) 
		GetTimestep(&dt);
		t += dt; 
		#elif !defined(__DPRK5) && !defined(__ADAPTIVE_STEP)
		t = iters * dt;
		#endif

		// Check System: Determine if system has blown up or integration limits reached
		SystemCheck(dt, iters);
	}
	//////////////////////////////
	// End Integration
	//////////////////////////////
	

 	// ------------------------------- 
	// Final Writes to Output File
	// -------------------------------
	FinalWriteAndCloseOutputFile(N, iters, save_data_indx);
	
	// -------------------------------
	// Clean Up 
	// -------------------------------
	FreeMemory(RK_data);
}
/**
 * Function to perform a single step of the RK5 or Dormand Prince scheme
 * @param dt       The current timestep of the system
 * @param N        Array containing the dimensions of the system
 * @param local_Nx Int indicating the local size of the first dimension of the arrays	
 * @param RK_data  Struct pointing the Integration variables: stages, tmp arrays, rhs and arrays needed for NonlinearRHS function
 */
#if defined(__RK5) || defined(__DPRK5)
void RK5DPStep(const double dt, const long int* N, const int iters, const ptrdiff_t local_Nx, RK_data_struct* RK_data) {

	// Initialize vairables
	int tmp1;
	int indx;
	#if defined(__VISCOUS)
	double k_sqr;
	double D_fac;
	#endif
	const long int Ny  		  = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;
	#if defined(__DPRK5)
	const long int Nx = N[0];
	double dp_ho_step;
	double err_sum;
	double err_denom;
	#endif
	#if defined(PHASE_ONLY)
	// Pre-record the amplitudes so they can be reset after update step
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// record amplitudes
			run_data->tmp_a_k[indx] = cabs(run_data->psi_hat[indx]);
		}
	}
	double tmp_a_k_norm;
	#endif
	
	/////////////////////
	/// RK STAGES
	/////////////////////
	// ----------------------- Stage 1
	NonlinearRHS(run_data->psi_hat, RK_data->RK1, RK_data->grad_psi_hat, RK_data->grad_psi);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp1 + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->psi_hat[indx] + dt * RK5_A21 * RK_data->RK1[indx];
		}
	}
	// ----------------------- Stage 2
	NonlinearRHS(RK_data->RK_tmp, RK_data->RK2, RK_data->grad_psi_hat, RK_data->grad_psi);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp1 + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->psi_hat[indx] + dt * RK5_A31 * RK_data->RK1[indx] + dt * RK5_A32 * RK_data->RK2[indx];
		}
	}
	// ----------------------- Stage 3
	NonlinearRHS(RK_data->RK_tmp, RK_data->RK3, RK_data->grad_psi_hat, RK_data->grad_psi);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp1 + j;

			// Update temporary input for nonlinear termiters, 
			RK_data->RK_tmp[indx] = run_data->psi_hat[indx] + dt * RK5_A41 * RK_data->RK1[indx] + dt * RK5_A42 * RK_data->RK2[indx] + dt * RK5_A43 * RK_data->RK3[indx];
		}
	}
	// ----------------------- Stage 4
	NonlinearRHS(RK_data->RK_tmp, RK_data->RK4, RK_data->grad_psi_hat, RK_data->grad_psi);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp1 + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->psi_hat[indx] + dt * RK5_A51 * RK_data->RK1[indx] + dt * RK5_A52 * RK_data->RK2[indx] + dt * RK5_A53 * RK_data->RK3[indx] + dt * RK5_A54 * RK_data->RK4[indx];
		}
	}
	// ----------------------- Stage 5
	NonlinearRHS(RK_data->RK_tmp, RK_data->RK5, RK_data->grad_psi_hat, RK_data->grad_psi);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp1 + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->psi_hat[indx] + dt * RK5_A61 * RK_data->RK1[indx] + dt * RK5_A62 * RK_data->RK2[indx] + dt * RK5_A63 * RK_data->RK3[indx] + dt * RK5_A64 * RK_data->RK4[indx] + dt * RK5_A65 * RK_data->RK5[indx];
		}
	}
	// ----------------------- Stage 6
	NonlinearRHS(RK_data->RK_tmp, RK_data->RK6, RK_data->grad_psi_hat, RK_data->grad_psi);
	#if defined(__DPRK5)
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp1 + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->psi_hat[indx] + dt * RK5_A71 * RK_data->RK1[indx] + dt * RK5_A73 * RK_data->RK3[indx] + dt * RK5_A74 * RK_data->RK4[indx] + dt * RK5_A75 * RK_data->RK5[indx] + dt * RK5_A76 * RK_data->RK6[indx];
		}
	}
	// ----------------------- Stage 7
	NonlinearRHS(RK_data->RK_tmp, RK_data->RK7, RK_data->grad_psi_hat, RK_data->grad_psi);
	#endif

	/////////////////////
	/// UPDATE STEP
	/////////////////////
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp1 + j;

			//---------- Pre-record the amplitudes
			#if defined(PHASE_ONLY)
			tmp_a_k_norm = cabs(run_data->psi_hat[indx]);
			#endif

			//---------- Update Fourier vorticity with the RHS
			#if defined(__INVISCID)
			// Update the Fourier space vorticity with the RHS
			run_data->psi_hat[indx] = run_data->psi_hat[indx] + (dt * (RK5_B1 * RK_data->RK1[indx]) + dt * (RK5_B3 * RK_data->RK3[indx]) + dt * (RK5_B4 * RK_data->RK4[indx]) + dt * (RK5_B5 * RK_data->RK5[indx]) + dt * (RK5_B6 * RK_data->RK6[indx]));
			
			#elif defined(__VISCOUS)
			// Compute the pre factors for the RK4CN update step
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			
			#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// Both Hyperviscosity and Ekman drag
			D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_POW)); 
			#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// No hyperviscosity but we have Ekman drag
			D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_POW)); 
			#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
			// Hyperviscosity only
			D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->VIS_POW)); 
			#else 
			// No hyper viscosity or no ekman drag -> just normal viscosity
			D_fac = dt * (sys_vars->NU * k_sqr); 
			#endif

			// Complete the update step
			run_data->psi_hat[indx] = run_data->psi_hat[indx] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK5_B1 * RK_data->RK1[indx] + RK5_B3 * RK_data->RK3[indx] + RK5_B4 * RK_data->RK4[indx] + RK5_B5 * RK_data->RK5[indx] + RK5_B6 * RK_data->RK6[indx]);
			#endif
			
			//---------- Reset the Fourier amplitudes
			#if defined(PHASE_ONLY)
			run_data->psi_hat[indx] *= (tmp_a_k_norm / cabs(run_data->psi_hat[indx]));
			#endif

			//---------- Compute the Dormand Prince Step Error
			#if defined(__DPRK5)
			if (iters > 1) {
				// Get the higher order update step
				dp_ho_step = run_data->psi_hat[indx] + (dt * (RK5_Bs1 * RK_data->RK1[indx]) + dt * (RK5_Bs3 * RK_data->RK3[indx]) + dt * (RK5_Bs4 * RK_data->RK4[indx]) + dt * (RK5_Bs5 * RK_data->RK5[indx]) + dt * (RK5_Bs6 * RK_data->RK6[indx]) + dt * (RK5_Bs7 * RK_data->RK7[indx]));
				
				// Denominator in the error
				err_denom = DP_ABS_TOL + DPMax(cabs(RK_data->psi_hat_last[indx]), cabs(run_data->psi_hat[indx])) * DP_REL_TOL;

				// Compute the sum for the error
				err_sum += pow((run_data->psi_hat[indx] - dp_ho_step) / err_denom, 2.0);
			}
			#endif
		}
	}
	#if defined(__DPRK5)
	if (iters > 1) {
		// Reduce and sync the error sum across the processes
		MPI_Allreduce(MPI_IN_PLACE, &err_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Compute the error
		RK_data->DP_err = sqrt(1.0 / (Nx * Ny_Fourier) * err_sum);

		// Record the Fourier velocity for the next step
		for (int i = 0; i < local_Nx; ++i) {
			tmp1 = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {			
				indx = tmp1 + j;

				// Record the vorticity
				RK_data->psi_hat_last[indx] = run_data->psi_hat[indx];
			}
		}
	}
	#endif
}
#endif
#if defined(__DPRK5)
/**
 * Function used to find the max between two numbers -> used in the Dormand Prince scheme
 * @param  a Double that will be used to find the max
 * @param  b Double that will be used to find the max
 * @return   The max between the two inputs
 */
double DPMax(double a, double b) {

	// Initailize max
	double max;

	// Check Max
	if (a > b) {
		max = a;
	}
	else {
		max = b;
	}

	// Return max
	return max;
}
/**
 * Function used to find the min between two numbers
 * @param  a Double that will be used to find the min
 * @param  b Double that will be used to find the min
 * @return   The minimum of the two inputs
 */
double DPMin(double a, double b) {

	// Initialize min
	double min;

	if (a < b) {
		min = a;
	}
	else {
		min = b;
	}

	// return the result
	return min;
}
#endif
/**
 * Function to perform one step using the 4th order Runge-Kutta method
 * @param dt       The current timestep of the system
 * @param N        Array containing the dimensions of the system
 * @param local_Nx Int indicating the local size of the first dimension of the arrays	
 * @param RK_data  Struct pointing the Integration variables: stages, tmp arrays, rhs and arrays needed for NonlinearRHS function
 */
#if defined(__RK4)
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Nx, RK_data_struct* RK_data) {

	// Initialize vairables
	int tmp;
	int indx;
	#if defined(__VISCOUS)
	double k_sqr;
	double D_fac;
	#endif
	const long int Ny_Fourier = N[1] / 2 + 1;
	#if defined(PHASE_ONLY)
	// Pre-record the amplitudes so they can be reset after update step
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// record amplitudes
			run_data->tmp_a_k[indx] = cabs(run_data->psi_hat[indx]);
		}
	}
	double tmp_a_k_norm;
	#endif


	/////////////////////
	/// RK STAGES
	/////////////////////
	// ----------------------- Stage 1
	NonlinearRHS(run_data->psi_hat, RK_data->RK1, RK_data->grad_psi_hat, RK_data->grad_psi);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->psi_hat[indx] + dt * RK4_A21 * RK_data->RK1[indx];
		}
	}
	// ----------------------- Stage 2
	NonlinearRHS(RK_data->RK_tmp, RK_data->RK2, RK_data->grad_psi_hat, RK_data->grad_psi);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->psi_hat[indx] + dt * RK4_A32 * RK_data->RK2[indx];
		}
	}
	// ----------------------- Stage 3
	NonlinearRHS(RK_data->RK_tmp, RK_data->RK3, RK_data->grad_psi_hat, RK_data->grad_psi);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Update temporary input for nonlinear term
			RK_data->RK_tmp[indx] = run_data->psi_hat[indx] + dt * RK4_A43 * RK_data->RK3[indx];
		}
	}
	// ----------------------- Stage 4
	NonlinearRHS(RK_data->RK_tmp, RK_data->RK4, RK_data->grad_psi_hat, RK_data->grad_psi);
	
	
	/////////////////////
	/// UPDATE STEP
	/////////////////////
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			//---------- Pre-record the amplitudes
			#if defined(PHASE_ONLY)
			tmp_a_k_norm = cabs(run_data->psi_hat[indx]);
			#endif

			//---------- Update Fourier vorticity with the RHS
			#if defined(__INVISCID)
			// The inviscid case
			run_data->psi_hat[indx] = run_data->psi_hat[indx] + (dt * (RK4_B1 * RK_data->RK1[indx]) + dt * (RK4_B2 * RK_data->RK2[indx]) + dt * (RK4_B3 * RK_data->RK3[indx]) + dt * (RK4_B4 * RK_data->RK4[indx]));
			#elif defined(__VISCOUS)
			// Compute the pre factors for the RK4CN update step
			k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);
			
			#if defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// Both Hyperviscosity and Ekman drag
			D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_POW)); 
			#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) 
			// No hyperviscosity but we have Ekman drag
			D_fac = dt * (sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_POW)); 
			#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) 
			// Hyperviscosity only
			D_fac = dt * (sys_vars->NU * pow(k_sqr, sys_vars->VIS_POW)); 
			#else 
			// No hyper viscosity or no ekman drag -> just normal viscosity
			D_fac = dt * (sys_vars->NU * k_sqr); 
			#endif

			// Update Fourier velocity potentials of the viscous case
			run_data->psi_hat[indx] = run_data->psi_hat[indx] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK4_B1 * RK_data->RK1[indx] + RK4_B2 * RK_data->RK2[indx] + RK4_B3 * RK_data->RK3[indx] + RK4_B4 * RK_data->RK4[indx]);
			#endif

			//---------- Reset the Fourier amplitudes
			#if defined(PHASE_ONLY)
			run_data->psi_hat[indx] *= (tmp_a_k_norm / cabs(run_data->psi_hat[indx]));
			#endif
		}
	}
	#if defined(__NONLIN)
	// Record the nonlinear term with the updated Fourier vorticity
	NonlinearRHS(run_data->psi_hat, run_data->nonlinterm, RK_data->grad_psi_hat, RK_data->grad_psi);
	#endif
}
#endif
/**
 * Computes the nonlinear term of the RHS -> first computes the gradients in Fourier space and then transfroms to real space
 * where the multiplication is performed before being transformed back to Fourier space to finish computing the nonlinear term
 * @param psi_hat      The current velocity potential 
 * @param nonlin_term  The output: the result of computing the nonlinear term
 * @param grad_psi_hat Batch array to hold the derivatives in Fourier space
 * @param grad_psi     Batch array to hold the derivatives in real space
 */
void NonlinearRHS(fftw_complex* psi_hat, fftw_complex* nonlin_term, fftw_complex* grad_psi_hat, double* grad_psi) {

	// Initialize variables
	int tmp, indx;
	const ptrdiff_t local_Nx  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	fftw_complex k_sqr;
	double grad_psi_x, grad_psi_y;
	double norm_fac = 1.0 / (Nx * Ny);

	// -----------------------------------
	// Compute Fourier Space Derivatives
	// -----------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * (Ny_Fourier);
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Compute the derivatives in Fourier space
			grad_psi_hat[SYS_DIM * indx + 0] = I * run_data->k[0][i] * psi_hat[indx];
			grad_psi_hat[SYS_DIM * indx + 1] = I * run_data->k[1][j] * psi_hat[indx];
		}
	}

	// ----------------------------------
	// Transform to Real Space
	// ----------------------------------
	// Batch transform both fourier velocites to real space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), grad_psi_hat, grad_psi);

	// ----------------------------------
	// Multiply in Real Space
	// ----------------------------------
	// Square the real space gradients
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j; 

			// Get the current gradients
			grad_psi_x = grad_psi[SYS_DIM * indx + 0];
			grad_psi_y = grad_psi[SYS_DIM * indx + 1];

			// Square the gradients in real space
			grad_psi[SYS_DIM * indx + 0] = grad_psi_x * grad_psi_x;
			grad_psi[SYS_DIM * indx + 1] = grad_psi_y * grad_psi_y;
		}
	}

	// ----------------------------------
	// Transform Back to Fourier Space
	// ----------------------------------
	// Transform the squared gradients back to Fourier space, normalize and add together
	fftw_mpi_execute_dft_r2c((sys_vars->fftw_2d_dft_batch_r2c), grad_psi, grad_psi_hat);
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * (Ny_Fourier);
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			// Compute the nonlinear term
			nonlin_term[indx] = 0.5 * norm_fac * (grad_psi_hat[SYS_DIM * indx + 0] + grad_psi_hat[SYS_DIM * indx + 1]);
		}
	}

	// ----------------------------------
	// Apply Dealiasing
	// ----------------------------------
	ApplyDealiasing(nonlin_term, 1, sys_vars->N);
}
/**
 * Function to initialize the Real space collocation points arrays and Fourier wavenumber arrays
 * 
 * @param x Array containing the collocation points in real space
 * @param k Array to contain the wavenumbers on both directions
 * @param N Array containging the dimensions of the system
 */
void InitializeSpaceVariables(double** x, int** k, const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1;

	// Initialize local variables 
	ptrdiff_t local_Nx       = sys_vars->local_Nx;
	ptrdiff_t local_Nx_start = sys_vars->local_Nx_start;
	
	// Set the spatial increments
	sys_vars->dx = 2.0 * M_PI / (double )Nx;
	sys_vars->dy = 2.0 * M_PI / (double )Ny;

	// -------------------------------
	// Fill the first dirction 
	// -------------------------------
	int j = 0;
	for (int i = 0; i < Nx; ++i) {
		if((i >= local_Nx_start) && ( i < local_Nx_start + local_Nx)) { // Ensure each process only writes to its local array slice
			x[0][j] = (double) i * 2.0 * M_PI / (double) Nx;
			j++;
		}
	}
	j = 0;
	for (int i = 0; i < local_Nx; ++i) {
		if (local_Nx_start + i <= Nx / 2) {   // Set the first half of array to the positive k
			k[0][j] = local_Nx_start + i;
			j++;
		}
		else if (local_Nx_start + i > Nx / 2) { // Set the second half of array to the negative k
			k[0][j] = local_Nx_start + i - Nx;
			j++;
		}
	}

	// -------------------------------
	// Fill the second direction 
	// -------------------------------
	for (int i = 0; i < Ny; ++i) {
		if (i < Ny_Fourier) {
			k[1][i] = i;
		}
		x[1][i] = (double) i * 2.0 * M_PI / (double) Ny;
	}
}
/**
 * Function to compute the initial condition for the integration
 * @param psi     Real space velocity potential
 * @param psi_hat Fourier space velocitpotential
 * @param N       Array containing the dimensions of the system
 */
void InitialConditions(double* psi, fftw_complex* psi_hat, const long int* N, double* norm) {

	// Initialize variables
	int tmp, indx;
	const long int Nx         = N[0];
	const long int Ny 		  = N[1];
	const long int Ny_Fourier = N[1] / 2 + 1; 

	// Initialize local variables 
	ptrdiff_t local_Nx = sys_vars->local_Nx;

    // ------------------------------------------------
    // Set Seed for RNG
    // ------------------------------------------------
    srand(123456789);

	if(!(strcmp(sys_vars->u0, "HOPF_COLE"))) {
		// ------------------------------------------------
		// Taylor Green Initial Condition - Real Space
		// ------------------------------------------------
		double theta;
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = (tmp + j);

				// Get the initial condition that solves the Diffusion equation
				theta = cos(KAPPA * run_data->x[0][i]) * sin(KAPPA * run_data->x[1][j]) * exp(-2.0 * sys_vars->NU * 0.0) + 2.0;

				// Fill the velocity potential
				psi[indx] = 2.0 * sys_vars->NU * log(theta);
			}
		}

		// Transform velocities to Fourier space & dealias
		fftw_mpi_execute_dft_r2c(sys_vars->fftw_2d_dft_r2c, psi, psi_hat);
	}
	else if(!(strcmp(sys_vars->u0, "TG_VEL"))) {
		// ------------------------------------------------
		// Taylor Green Initial Condition - Real Space
		// ------------------------------------------------
		for (int i = 0; i < local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = (tmp + j);

				// Fill the velocities
				psi[indx] = sin(KAPPA * run_data->x[0][i]) * cos(KAPPA * run_data->x[1][j]);
			}
		}

		// Transform velocities to Fourier space & dealias
		fftw_mpi_execute_dft_r2c(sys_vars->fftw_2d_dft_r2c, psi, psi_hat);
	}
	else if (!(strcmp(sys_vars->u0, "RANDOM"))) {
		// ---------------------------------------
		// Random Initial Conditions
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Fill fourier space velocity
				psi_hat[indx] = ((double)rand() / (double) RAND_MAX) * cexp(((double)rand() / (double) RAND_MAX)* 2.0 * M_PI * I);
			}
		}		
	}
	else if (!(strcmp(sys_vars->u0, "TESTING"))) {
		// Initialize temp variables
		double inv_k_sqr;

		// ---------------------------------------
		// Powerlaw Amplitude & Fixed Phase
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				if ((run_data->k[0][i] == 0) && (run_data->k[1][j] == 0)){
					// Fill zero modes
					psi_hat[indx] = 0.0 + 0.0 * I;
				}
				else if (j == 0 && run_data->k[0][i] < 0 ) {
					// Amplitudes
					inv_k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Fill vorticity - this is for the kx axis - to enfore conjugate symmetry
					psi_hat[indx] = inv_k_sqr * cexp(-I * M_PI / 4.0);
				}
				else {
					// Amplitudes
					inv_k_sqr = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

					// Fill vorticity - fill the rest of the modes
					psi_hat[indx] = inv_k_sqr * cexp(I * M_PI / 4.0);
				}
			}
		}
	}
	else {
		printf("\n["MAGENTA"WARNING"RESET"] --- No initial conditions specified\n---> Using random initial conditions...\n");
		// ---------------------------------------
		// Random Initial Conditions
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp = i * (Ny_Fourier);
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				// Fill vorticity
				psi_hat[indx] = ((double)rand() / (double) RAND_MAX) * cexp(((double)rand() / (double) RAND_MAX) * 2.0 * M_PI * I);
			}
		}		
	}
	// -------------------------------------------------
	// Initialize the Dealiasing
	// -------------------------------------------------
	ApplyDealiasing(psi_hat, 1, N);
    
    
	// -------------------------------------------------
	// Initialize the Forcing
	// -------------------------------------------------
	// ApplyForcing(psi_hat, N);
	
	// -------------------------------------------------
	// Initialize Exact Soln 
	// -------------------------------------------------
	// If testing is enabled and Hopf-Cole initial condition selected -> compute exact solution for writing to file @ t = t0
	#if defined(TESTING)
	if(!(strcmp(sys_vars->u0, "HOPF_COLE"))) {
		ExactSoln(0.0, norm);
	}
	#endif
}
/**
 * Function to apply the selected dealiasing filter to the input array. Can be Fourier vorticity or velocity
 * @param array    	The array containing the Fourier modes to dealiased
 * @param array_dim The extra array dimension -> will be 1 for scalar or 2 for vector
 * @param N        	Array containing the dimensions of the system
 */
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N) {

	// Initialize variables
	int tmp, indx;
	ptrdiff_t local_Nx        = sys_vars->local_Nx;
	const long int Nx         = N[0];
	const long int Ny         = N[1];
	const long int Ny_Fourier = Ny / 2 + 1;
	double dealiased_k_sqr = pow(Nx / 3.0, 2.0);
	#if defined(__DEALIAS_HOU_LI)
	double hou_li_filter;
	#endif

	// --------------------------------------------
	// Apply Appropriate Filter 
	// --------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = array_dim * (tmp + j);

			#if defined(__DEALIAS_23)
			if ((double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) > dealiased_k_sqr) {
				for (int l = 0; l < array_dim; ++l) {
					// Set dealised modes to 0
					array[indx + l] = 0.0 + 0.0 * I;	
				}
			}
			else {
				continue;			
			}
			#elif __DEALIAS_HOU_LI
			// Compute Hou-Li filter
			hou_li_filter = exp(-36.0 * pow((sqrt(pow(run_data->k[0][i] / (Nx / 2), 2.0) + pow(run_data->k[1][j] / (Ny / 2), 2.0))), 36.0));

			for (int l = 0; l < array_dim; ++l) {
				// Apply filter and DFT normaliztion
				array[indx + l] *= hou_li_filter;
			}
			#endif
		}
	}
}
/**
 * Computes the exact solution via the Hopf-Cole transformation
 * @param t The current time in the simulation
 */
void ExactSoln(double t, double* norm) {

	// Initialize variables
	int tmp;
	int indx;
	double theta;
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];
	double norm_fac = 1.0 / (Nx * Ny);

	// Initialize the norms
	norm[0] = 0.0;
	norm[1] = 0.0;
	double l2_norm = 0.0;

	// Transform to Real space
	fftw_mpi_execute_dft_c2r(sys_vars->fftw_2d_dft_c2r, run_data->psi_hat, run_data->psi);

	// -------------------------------
	// Compute The Exact Solution
	// -------------------------------
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			// Get the solution to the Heat Equation 
			theta = cos(KAPPA * run_data->x[0][i]) * sin(KAPPA * run_data->x[1][j]) * exp(-2.0 * sys_vars->NU * t) + 2.0;

			// Get the solution in terms of velocity potential
			run_data->exact_soln[indx] = 2.0 * sys_vars->NU * log(theta);

			// Update the norms
			norm[0] = fmax(fabs(run_data->psi[indx] * norm_fac - run_data->exact_soln[indx]), norm[0]);
			l2_norm += pow(run_data->psi[indx] * norm_fac - run_data->exact_soln[indx], 2.0);
		}
	}

	// Normalize the L2 norm
	norm[1] = sqrt(l2_norm * norm_fac);
}
/**
 * Function to update the timestep if adaptive timestepping is enabled
 * @param dt The current timestep
 */
void GetTimestep(double* dt) {

	// -------------------------------
	// Compute New Timestep
	// -------------------------------
	#if defined(__CFL_STEP)
	// Initialize variables
	int tmp;
	int indx;
	double norm_fac = 1.0 / (double)(Nx * Ny);
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;

	// ---------------------------- Get the Real Space Velocities
	// Compute the velocity in Fourier space
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * Ny_Fourier;
		for (int j = 0; j < Ny_Fourier; ++j) {
			indx = tmp + j;

			run_data->u_hat[SYS_DIM * indx + 0] = I * run_data->k[0][i] * run_data->psi_hat[indx];
			run_data->u_hat[SYS_DIM * indx + 1] = I * run_data->k[1][j] * run_data->psi_hat[indx];
		}
	}

	// Transform back to Fourier space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), run_data->u_hat, run_data->u);
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			// Normalize
			run_data->u[SYS_DIM * indx + 0] *= norm_fac;
			run_data->u[SYS_DIM * indx + 1] *= norm_fac;
		}
	}

	// ------------------------------- Find the Times Scales 
	// Find the convective scales
	double scales_convect = 0.0;
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (sys_vars->N[1] + 2);
		for (int j = 0; j < sys_vars->N[1]; ++j) {
			indx = tmp + j;

			scales_convect = fmax(M_PI * (fabs(run_data->u[SYS_DIM * indx + 0]) / sys_vars->dx + fabs(run_data->u[SYS_DIM * indx + 1]) / sys_vars->dy), scales_convect);
		}
	}
	// Gather the maximum of the convective scales
	MPI_Allreduce(MPI_IN_PLACE, &scales_convect, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	// Find the diffusive scales
	double scales_diff = pow(M_PI, 2.0) * (sys_vars->NU / pow(sys_vars->dx, 2.0) + sys_vars->NU / pow(sys_vars->dy, 2.0));

	// ------------------------------- Update Timestep
	(* dt) = sys_vars->CFL_CONST / (scales_convect + scales_diff);
	#else
	double dt_new;
	double psi_max;

	// -------------------------------
	// Get Current Max Vorticity 
	// -------------------------------
	psi_max = GetMaxData("VEL_POT");

	// Find proposed timestep = h_0 * (max{psi_hat(0)} / max{psi_hat(t)}) -> this ensures that the maximum vorticity by the timestep is constant
	dt_new = (sys_vars->dt) * (sys_vars->psi_max_init / psi_max);	

	// Gather new timesteps from all processes -> pick the smallest one
	MPI_Allreduce(MPI_IN_PLACE, &dt_new, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	
	// -------------------------------
	// Update Timestep
	// -------------------------------
	// Check if new timestep meets approved criteria
	if (dt_new > 2.0 * (*dt)) {  
		// Timestep can be increased by no more than twice current timestep
		(*dt) = 2.0 * (*dt); 
	}
	else {
		// Update with new timestep - new timestep is checked for criteria in SystemsCheck() function 
		(*dt) = dt_new;
	}
	#endif

	// -------------------------------
	// Record Min/Max Timestep
	// -------------------------------
	#if !defined(__DPRK5)
	// Get min timestep
	sys_vars->min_dt = fmin((*dt), sys_vars->min_dt);
	// Get max timestep
	sys_vars->max_dt = fmax((*dt), sys_vars->max_dt);
	#endif
}
/**
 * Function used to compute of either the velocity or vorticity
 * @return  Returns the computed maximum
 */
double GetMaxData(char* dtype) {

	// Initialize variables
	const long int Nx         = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	int tmp;
	int indx;
	double norm_fac = 1.0 / (double )(Nx * Ny);

	// -------------------------------
	// Compute the Data 
	// -------------------------------
	// Get the Real Sapce Data
	if (strcmp(dtype, "VEL") == 0) {
		// Compute the velocity in Fourier space
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * Ny_Fourier;
			for (int j = 0; j < Ny_Fourier; ++j) {
				indx = tmp + j;

				run_data->u_hat[SYS_DIM * indx + 0] = I * run_data->k[0][i] * run_data->psi_hat[indx];
				run_data->u_hat[SYS_DIM * indx + 1] = I * run_data->k[1][j] * run_data->psi_hat[indx];
			}
		}

		// Transform back to Fourier space
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_batch_c2r), run_data->u_hat, run_data->u);
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->u[SYS_DIM * indx + 0] *= norm_fac;
				run_data->u[SYS_DIM * indx + 1] *= norm_fac;
			}
		}
	}
	else if (strcmp(dtype, "VEL_POT") == 0) {
		// Perform transform back to real space for the vorticity
		fftw_mpi_execute_dft_c2r((sys_vars->fftw_2d_dft_c2r), run_data->psi_hat, run_data->psi);
		for (int i = 0; i < sys_vars->local_Nx; ++i) {
			tmp = i * (Ny + 2);
			for (int j = 0; j < Ny; ++j) {
				indx = tmp + j;

				// Normalize
				run_data->psi[indx] *= norm_fac;
			}
		}
	}

	// -------------------------------
	// Find the maximum
	// -------------------------------
	// Define the maximum
	double max = 0.0;

	// Loop over array to find the maximum
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp = i * (Ny + 2);
		for (int j = 0; j < Ny; ++j) {
			indx = tmp + j;

			// Find the max of the Real Space velocity
			if (strcmp(dtype, "VEL") == 0) {
				for (int l = 0; l < SYS_DIM; ++l) {
					// Check if maximum
					max = fmax(fabs(run_data->u[SYS_DIM * indx + l]), max);
				}
			}
			// Find the max of the Real Space Velocity Potential
			else if (strcmp(dtype, "VEL_POT") == 0) {
				// Check if maximum
				max = fmax(fabs(run_data->psi_hat[indx]), max);
			}
		}			
	}

	// Now synchronize the maximum over all process
	MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	return max;
}
/**
 * Used to print update to screen
 * @param iters          The current iteration of the integration
 * @param t              The current time in the simulation
 * @param dt             The current timestep in the simulation
 * @param T              The final time of the simulation
 * @param save_data_indx The saving index for output data
 * @param norm           Array to hold the Linf and L2 norm of the error
 */
void PrintUpdateToTerminal(int iters, double t, double dt, double T, int save_data_indx, double* norm) {

	// Initialize variables
	double max_psi = GetMaxData("VEL_POT");

	if( !(sys_vars->rank) ) {	
	#if defined(TESTING)
		// Print to screen
		printf("Iter: %d/%ld\tt: %1.6lf/%1.3lf\tdt: %g\tMax Psi: %1.4lf\tKE: %g\tL2norm: %g\tLinf: %g\n", iters, sys_vars->num_t_steps, t, T, dt, max_psi, run_data->tot_energy[save_data_indx], norm[1], norm[0]);
	#else
		// Print to screen
		printf("Iter: %d/%ld\tt: %1.6lf/%1.3lf\tdt: %g\tMax Psi: %1.4lf\tKE: %g\n", iters, sys_vars->num_t_steps, t, T, dt, max_psi, run_data->tot_energy[save_data_indx]);
	#endif
	}
}
/**
 * Function that checks the system to see if it is ok to continue integrations. Checks for blow up, timestep and iteration limits etc
 * @param dt    The updated timestep for the next iteration
 * @param iters The number of iterations for the next iteration
 */
void SystemCheck(double dt, int iters) {

	// Initialize variables
	double psi_max;	

	// -------------------------------
	// Get Current Max Vorticity 
	// -------------------------------
	psi_max = GetMaxData("VEL_POT");

	// -------------------------------
	// Check Stopping Criteria 
	// -------------------------------
	if (psi_max >= MAX_PSI_LIM)	{
		fprintf(stderr, "\n["YELLOW"SOVLER FAILURE"RESET"] --- System has reached maximum Velocity Potential limt at Iter: ["CYAN"%d"RESET"] - Psi_max: ["CYAN"%g"RESET"] Psi_lim: ["CYAN"%g"RESET"]\n-->> Exiting!!!\n", iters, psi_max, MAX_PSI_LIM);
		exit(1);
	}
	else if (dt <= MIN_STEP_SIZE) {
		fprintf(stderr, "\n["YELLOW"SOVLER FAILURE"RESET"] --- Timestep has become too small to continue at Iter: ["CYAN"%d"RESET"] - dt: ["CYAN"%g"RESET"] dt_lim: ["CYAN"%g"RESET"]\n-->> Exiting!!!\n", iters, dt, MIN_STEP_SIZE);
		exit(1);		
	}
	else if (iters >= (int)MAX_ITERS) {
		fprintf(stderr, "\n["YELLOW"SOVLER FAILURE"RESET"] --- The maximum number of iterations has been reached at Iter: ["CYAN"%d"RESET"] - Iter: ["CYAN"%d"RESET"] Iter_lim: ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", iters, iters, (int)MAX_ITERS);
		exit(1);		
	}
}	
/**
* Function to compute the system measurables such as energy, enstrophy, palinstrophy, helicity, energy and enstrophy dissipation rates, and spectra at once on the local processes for the current timestep
* @param t 			The current time in the simulation
* @param iter 		The index in the system arrays for the current timestep
* @param RK_data 	Struct containing the integration varaiables needed for the nonlinear term function
*/
void ComputeSystemMeasurables(double t, int iter, RK_data_struct* RK_data) {

	// Initialize variables
	int tmp;
	int indx;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	double k_sqr, pre_fac;
	double norm_fac  = 0.5 / pow(Nx * Ny, 2.0);
    double const_fac = 4.0 * pow(M_PI, 2.0);
    #if defined(__ENRG_FLUX)
    double lwr_sbst_lim_sqr = pow(LWR_SBST_LIM, 2.0);
    double upr_sbst_lim_sqr = pow(UPR_SBST_LIM, 2.0);
    #endif

    // Record the initial time
    #if defined(__TIME) && !defined(TRANSIENTS)
    if (!(sys_vars->rank)) {
    	run_data->time[iter] = t;
    }
    #endif

    // If adaptive stepping check if within memory limits
    if (iter >= sys_vars->num_print_steps) {
    	// Print warning to screen if we have exceeded the memory limits for the system measurables arrays
    	printf("\n["MAGENTA"WARNING"RESET"] --- Unable to write system measures at Indx: [%d] t: [%lf] ---- Number of intergration steps is now greater then memory allocated\n", iter, t);
    }


    // ------------------------------------
    // Initialize Measurables
    // ------------------------------------
    #if defined(__SYS_MEASURES)
    // Initialize totals
    if (iter < sys_vars->num_print_steps) {
	    run_data->tot_energy[iter] = 0.0;
	    run_data->enrg_diss[iter]  = 0.0;
	    #if defined(__ENRG_FLUX)
		run_data->enrg_flux_sbst[iter] = 0.0;
		run_data->enrg_diss_sbst[iter] = 0.0;
		#endif
	}
    #endif 
    #if defined(__ENRG_SPECT) || defined(__ENST_SPECT)
    // Initialize spectra
    for (int i = 0; i < sys_vars->n_spect; ++i) {
    	#if defined(__ENRG_SPECT)
    	run_data->enrg_spect[i] = 0.0;
    	#endif
    	#if defined(__ENRG_FLUX_SPECT)
    	run_data->enrg_flux_spect[spec_indx] = 0.0;
    	#endif
    }
    #endif

    #if defined(__ENRG_FLUX) || defined(__ENRG_FLUX_SPECT)
    // Compute the nonlinear term
    NonlinearRHS(run_data->psi_hat, RK_data->RK1, RK_data->grad_psi_hat, RK_data->grad_psi);
    #endif

    // -------------------------------------
    // Compute Measurables in Fourier Space
    // -------------------------------------
    // Loop over Fourier vorticity
    for (int i = 0; i < local_Nx; ++i) {
    	tmp = i * Ny_Fourier;
    	for (int j = 0; j < Ny_Fourier; ++j) {
    		indx = tmp + j;

    		// The |k|^2 prefactor
    		k_sqr = (double )(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]);

    		///--------------------------------- System Measures
    		#if defined(__SYS_MEASURES)
	   		if (iter < sys_vars->num_print_steps) {
	   			// Get the appropriate prefactor
	   			#if defined(HYPER_VISC) && defined(EKMN_DRAG)  // Both Hyperviscosity and Ekman drag
	   			pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->VIS_POW) + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_POW);
	   			#elif !defined(HYPER_VISC) && defined(EKMN_DRAG) // No hyperviscosity but we have Ekman drag
	   			pre_fac = sys_vars->NU * k_sqr + sys_vars->EKMN_ALPHA * pow(k_sqr, sys_vars->EKMN_POW);
	   			#elif defined(HYPER_VISC) && !defined(EKMN_DRAG) // Hyperviscosity only
	   			pre_fac = sys_vars->NU * pow(k_sqr, sys_vars->VIS_POW);
	   			#else // No hyper viscosity or no ekman drag -> just normal viscosity
	   			pre_fac = sys_vars->NU * k_sqr; 
	   			#endif

	    		// Update the sums
	    		if ((j == 0) || (j == Ny_Fourier - 1)) { // only count the 0 and N/2 modes once as they have no conjugate
	    			run_data->enrg_diss[iter]  += k_sqr * k_sqr * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])); 
	    			run_data->tot_energy[iter] += k_sqr * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx]));
		    		#if defined(__ENRG_FLUX)
	    			if ((k_sqr >= lwr_sbst_lim_sqr) && (k_sqr < upr_sbst_lim_sqr)) { // define the subset to consider for the flux and dissipation
		    			run_data->enrg_flux_sbst[iter] += creal(run_data->psi_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->psi_hat[indx]) * RK_data->RK1[indx]) * k_sqr;
		    			run_data->enrg_diss_sbst[iter] += pre_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * k_sqr;
		    		}
		    		#endif
	    		}
	    		else {
	    			run_data->enrg_diss[iter]  += 2.0 * k_sqr * k_sqr * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])); 
	    			run_data->tot_energy[iter] += 2.0 * k_sqr * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx]));
	    			#if defined(__ENRG_FLUX)
	    			if ((k_sqr >= lwr_sbst_lim_sqr) && (k_sqr < upr_sbst_lim_sqr)) { // define the subset to consider for the flux and dissipation
	    				run_data->enrg_flux_sbst[iter] += 2.0 * creal(run_data->psi_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->psi_hat[indx]) * RK_data->RK1[indx]) * k_sqr;
	    				run_data->enrg_diss_sbst[iter] += 2.0 * pre_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * k_sqr;
	    			}
	    			#endif
	    		}
	    		#endif
	    	}

    		///--------------------------------- Spectra
    		#if defined(__ENRG_SPEC) || defined(__ENRG_FLUX_SPECT)
    		// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
    		spec_indx = (int) round( sqrt( (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j]) ) );

    		if ((j == 0) || (j == Ny_Fourier - 1)) {
    			// Update the current bin
    			#if defined(__ENRG_SPEC)
    			run_data->enrg_spect[spec_indx] += const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * k_sqr;
    			#endif
    			#if defined(__ENRG_FLUX_SPECT)
    			run_data->enrg_flux_spect[spec_indx] += const_fac * norm_fac * creal(run_data->psi_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->psi_hat[indx]) * RK_data->RK1[indx]) * k_sqr;
    			#endif
    		}
    		else {
    			// Update the spectra sums for the current mode
    			#if defined(__ENRG_SPEC)
    			run_data->enrg_spect[spec_indx] += 2.0 * const_fac * norm_fac * cabs(run_data->psi_hat[indx] * conj(run_data->psi_hat[indx])) * k_sqr;
    			#endif
    			#if defined(__ENRG_FLUX_SPECT)
    			run_data->enrg_flux_spect[spec_indx] += 2.0 * const_fac * norm_fac * creal(run_data->psi_hat[indx] * conj(RK_data->RK1[indx]) + conj(run_data->psi_hat[indx]) * RK_data->RK1[indx]) * k_sqr;
    			#endif
    		}
    		#endif
    	}
    }
    // ------------------------------------
    // Normalize Measureables 
    // ------------------------------------	
    #if defined(__SYS_MEASURES)
    if (iter < sys_vars->num_print_steps) {
	    // Normalize results and take into account computation in Fourier space
	    run_data->enrg_diss[iter]  *= 2.0 * const_fac * norm_fac;
	    run_data->tot_energy[iter] *= const_fac * norm_fac;
	}
    #endif
    #if defined(__ENRG_FLUX)
    run_data->enrg_flux_sbst[iter] *= const_fac * norm_fac;
    run_data->enrg_diss_sbst[iter] *= 2.0 * const_fac * norm_fac;
    #endif
}
/**
 * Function to initialize and compute the system measurables and spectra of the initial conditions
 * @param RK_data The struct containing the Runge-Kutta arrays to compute the nonlinear term for the fluxes
 */
void InitializeSystemMeasurables(RK_data_struct* RK_data) {

	// Set the size of the arrays to twice the number of printing steps to account for extra steps due to adaptive stepping
	#if defined(__ADAPTIVE_STEP)
	sys_vars->num_print_steps = 2 * sys_vars->num_print_steps;
	#else
	sys_vars->num_print_steps = sys_vars->num_print_steps;
	#endif
	int print_steps = sys_vars->num_print_steps;

	// Get the size of the spectrum arrays
	#if defined(__ENRG_SPECT) || defined(__ENRG_FLUX_SPECT)
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];

	sys_vars->n_spect = (int) sqrt(pow((double)Nx / 2.0, 2.0) + pow((double)Ny / 2.0, 2.0)) + 1;
	int n_spect = sys_vars->n_spect;
	#endif
		
	// ------------------------
	// Allocate Memory
	// ------------------------
	//---------------------------- Allocate memory for the system measures
	#if defined(__SYS_MEASURES)
	// Total Energy in the system
	run_data->tot_energy = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_energy == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Energy");
		exit(1);
	}	

	// Total Enstrophy
	run_data->tot_enstr = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_enstr == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Enstrophy");
		exit(1);
	}	

	// Total Palinstrophy
	run_data->tot_palin = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_palin == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Palinstrophy");
		exit(1);
	}	

	// Energy Dissipation Rate
	run_data->enrg_diss = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_diss == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Rate");
		exit(1);
	}	

	// Enstrophy Dissipation Rate
	run_data->enst_diss = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_diss == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Rate");
		exit(1);
	}	
	#endif

	//---------------------------- Allocate memory for the Energy Spectrum
	#if defined(__ENRG_SPECT )
	// Energy Spectrum
	run_data->enrg_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Spectrum");
		exit(1);
	}	
	#endif

	//---------------------------- Allocate memory for the Energy Flux
	#if defined(__ENRG_FLUX)
	// Energy Flux
	run_data->enrg_flux_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_flux_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Flux Subset");
		exit(1);
	}	

	// Energy Dissipation Rate
	run_data->enrg_diss_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_diss_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Rate Subset");
		exit(1);
	}
	#endif

	//---------------------------- Allocate memory for the Energy Flux Spectra
	#if defined(__ENRG_FLUX_SPECT )
	// Energy Spectrum
	run_data->enrg_flux_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_flux_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Flux Spectrum");
		exit(1);
	}	
	#endif

	//---------------------------- Allocate memory for the Time array
	#if defined(__TIME)
	if (!(sys_vars->rank)){
		run_data->time = (double* )fftw_malloc(sizeof(double) * print_steps);
		if (run_data->time == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time");
			exit(1);
		}	
	}
	#endif


	// ----------------------------
	// Get Measurables of the ICs
	// ----------------------------
	// Compute the system measurables, flux and spectra on the initial condition
	ComputeSystemMeasurables(sys_vars->t0, 0, RK_data);
}	
/**
 * Function to initialize all the integration time variables
 * @param t0           The initial time of the simulation
 * @param t            The current time of the simulaiton
 * @param dt           The timestep
 * @param T            The final time of the simulation
 * @param trans_steps  The number of iterations to perform before saving to file begins
 */
void InitializeIntegrationVariables(double* t0, double* t, double* dt, double* T, long int* trans_steps) {
	
	// -------------------------------
	// Get the Timestep
	// -------------------------------
	#if defined(__ADAPTIVE_STEP)
	GetTimestep(&(sys_vars->dt));
	#endif

	// -------------------------------
	// Get Time variables
	// -------------------------------
	// Compute integration time variables
	(*t0) = sys_vars->t0;
	(*t ) = sys_vars->t0;
	(*dt) = sys_vars->dt;
	(*T ) = sys_vars->T;
	sys_vars->min_dt = 10;
	sys_vars->max_dt = MIN_STEP_SIZE;

	// -------------------------------
	// Integration Counters
	// -------------------------------
	// Number of time steps and saving steps
	sys_vars->num_t_steps = ((*T) - (*t0)) / (*dt);
	#if defined(TRANSIENTS)
	// Get the transient iterations
	(* trans_steps)       = (long int)(TRANS_FRAC * sys_vars->num_t_steps);
	sys_vars->trans_iters = (* trans_steps);

	// Get the number of steps to perform before printing to file -> allowing for a transient fraction of these to be ignored
	sys_vars->num_print_steps = (sys_vars->num_t_steps >= sys_vars->SAVE_EVERY ) ? (sys_vars->num_t_steps - sys_vars->trans_iters) / sys_vars->SAVE_EVERY : sys_vars->num_t_steps - sys_vars->trans_iters;	 
	if (!(sys_vars->rank)){
		printf("Total Iters: %ld\t Saving Iters: %ld\t Transient Steps: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps, sys_vars->trans_iters);
	}
	#else
	// Get the transient iterations
	(* trans_steps)       = 0;
	sys_vars->trans_iters = (* trans_steps);

	// Get the number of steps to perform before printing to file
	sys_vars->num_print_steps = (sys_vars->num_t_steps >= sys_vars->SAVE_EVERY ) ? sys_vars->num_t_steps / sys_vars->SAVE_EVERY + 1 : sys_vars->num_t_steps + 1; // plus one to include initial condition
	if (!(sys_vars->rank)){
		printf("Total Iters: %ld\t Saving Iters: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps);
	}
	#endif

	// Variable to control how ofter to print to screen -> set it to half the saving to file steps
	sys_vars->print_every = (sys_vars->num_t_steps >= 10 ) ? (int)sys_vars->SAVE_EVERY : 1;
}
/**
 * Wrapper function used to allocate memory all the nessecary local and global system and integration arrays
 * @param NBatch  Array holding the dimensions of the Fourier space arrays
 * @param RK_data Pointer to struct containing the integration arrays
 */
void AllocateMemory(const long int* NBatch, RK_data_struct* RK_data) {

	// Initialize variables
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;	

	// -------------------------------
	// Get Local Array Sizes - FFTW 
	// -------------------------------
	//  Find the size of memory for the FFTW transforms - use these to allocate appropriate memory
	sys_vars->alloc_local       = fftw_mpi_local_size_2d(Nx, Ny_Fourier, MPI_COMM_WORLD, &(sys_vars->local_Nx), &(sys_vars->local_Nx_start));
	sys_vars->alloc_local_batch = fftw_mpi_local_size_many((int)SYS_DIM, NBatch, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &(sys_vars->local_Nx), &(sys_vars->local_Nx_start));
	if (sys_vars->local_Nx == 0) {
		printf("\n["MAGENTA"WARNING"RESET"] --- FFTW was unable to allocate local memory for each process -->> Code will run but will be slow\n");
	}
	
	// -------------------------------
	// Allocate Space Variables 
	// -------------------------------
	// Allocate the wavenumber arrays
	run_data->k[0] = (int* )fftw_malloc(sizeof(int) * sys_vars->local_Nx);  // kx
	run_data->k[1] = (int* )fftw_malloc(sizeof(int) * Ny_Fourier);     		// ky
	if (run_data->k[0] == NULL || run_data->k[1] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "wavenumber list");
		exit(1);
	}

	// Allocate the collocation points
	run_data->x[0] = (double* )fftw_malloc(sizeof(double) * sys_vars->local_Nx);  // x direction 
	run_data->x[1] = (double* )fftw_malloc(sizeof(double) * Ny);     			  // y direction
	if (run_data->x[0] == NULL || run_data->x[1] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "collocation points");
		exit(1);
	}
	// -------------------------------
	// Allocate System Variables 
	// -------------------------------
	//---------------------------- Allocate the Real and Fourier space velocity potentials
	run_data->psi     = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local);
	if (run_data->psi == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Real Space Velocity Potential");
		exit(1);
	}
	run_data->psi_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (run_data->psi_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Space Velocity Potential");
		exit(1);
	}

	//---------------------------- Allocate the Real and Fourier space velociti
	run_data->u     = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (run_data->u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Real Space Velocities");
		exit(1);
	}
	run_data->u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (run_data->u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Space Velocities");
		exit(1);
	}

	//---------------------------- Allocate the Phase Only Array
	#if defined(PHASE_ONLY)
	// Allocate array for the Fourier amplitudes
	run_data->a_k = (double* )fftw_malloc(sizeof(double) * sys_vars->alloc_local_batch);
	if (run_data->a_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Amplitudes");
		exit(1);
	}
	// Allocate array for the Fourier phases
	run_data->phi_k = (double* )fftw_malloc(sizeof(double) * sys_vars->alloc_local_batch);
	if (run_data->phi_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Phases");
		exit(1);
	}
	// Allocate array for the Fourier phases
	run_data->tmp_a_k = (double* )fftw_malloc(sizeof(double) * sys_vars->alloc_local_batch);
	if (run_data->tmp_a_k == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Tmp Amplitudes");
		exit(1);
	}
	#endif

	//---------------------------- Allocate array for the exact solution
	#if defined(TESTING)
	run_data->exact_soln = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (run_data->exact_soln == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Taylor Green vortex solution");
		exit(1);
	}
	#endif

	//---------------------------- Allocate array for the nonlinear term
	#if defined(__NONLIN)
	// Allocate memory for recording the nonlinear term
	run_data->nonlinterm = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (run_data->nonlinterm == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Nonlinear Term");
		exit(1);
	}
	#endif

	// -------------------------------
	// Allocate Integration Variables 
	// -------------------------------
	// Runge-Kutta Integration arrays
	RK_data->grad_psi = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->grad_psi == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "nabla_psi");
		exit(1);
	}
	RK_data->grad_psi_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->grad_psi_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Nable_Psi_Hat");
		exit(1);
	}
	RK_data->RK1       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->RK1 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK1");
		exit(1);
	}
	RK_data->RK2       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->RK2 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK2");
		exit(1);
	}
	RK_data->RK3       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->RK3 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK3");
		exit(1);
	}
	RK_data->RK4       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->RK4 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK4");
		exit(1);
	}
	RK_data->RK_tmp    = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->RK_tmp == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK_tmp");
		exit(1);
	}
	#if defined(__RK5) || defined(__DPRK5)
	RK_data->RK5       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->RK5 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK5");
		exit(1);
	}
	RK_data->RK6       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->RK6 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK6");
		exit(1);
	}
	#endif
	#if defined(__DPRK5)
	RK_data->RK7       = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->RK7 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK7");
		exit(1);
	}
	RK_data->psi_hat_last = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local);
	if (RK_data->psi_hat_last == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "w_hat_last");
		exit(1);
	}
	#endif

	// -------------------------------
	// Initialize All Data 
	// -------------------------------
	int tmp_real, tmp_four;
	int indx_real, indx_four;
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp_real = i * (Ny + 2);
		tmp_four = i * Ny_Fourier;
		
		for (int j = 0; j < Ny; ++j){
			indx_real = tmp_real + j;
			indx_four = tmp_four + j;
			
			run_data->psi[indx_real]                    = 0.0;
			run_data->u[SYS_DIM * indx_real + 0]        = 0.0;
			run_data->u[SYS_DIM * indx_real + 1] 	  	= 0.0;
			RK_data->grad_psi[SYS_DIM * indx_real + 0] 	= 0.0;
			RK_data->grad_psi[SYS_DIM * indx_real + 1] 	= 0.0;
			#if defined(TESTING)
			run_data->exact_soln[indx_real]             = 0.0;
			#endif
			if (j < Ny_Fourier) {
				run_data->psi_hat[indx_four]			= 0.0;
				#if defined(PHASE_ONLY)
				run_data->a_k[indx_four] 				= 0.0;
				run_data->phi_k[indx_four]			    = 0.0;
				run_data->tmp_a_k[indx_four] 			= 0.0;
				#endif
				run_data->u_hat[SYS_DIM * indx_four + 0] 	   = 0.0 + 0.0 * I;
				run_data->u_hat[SYS_DIM * indx_four + 1] 	   = 0.0 + 0.0 * I;
				RK_data->RK_tmp[indx_four]    			 	   = 0.0 + 0.0 * I;
				RK_data->RK1[indx_four]                        = 0.0 + 0.0 * I;
				RK_data->RK2[indx_four]                        = 0.0 + 0.0 * I;
				RK_data->RK3[indx_four]                        = 0.0 + 0.0 * I;
				RK_data->RK4[indx_four]                        = 0.0 + 0.0 * I;
				RK_data->grad_psi_hat[SYS_DIM * indx_four + 0] = 0.0 + 0.0 * I;
				RK_data->grad_psi_hat[SYS_DIM * indx_four + 1] = 0.0 + 0.0 * I;
				#if defined(__NONLIN)
				run_data->nonlinterm[indx_four]	 = 0.0 + 0.0 * I;
				#endif
				#if defined(__RK5)
				RK_data->RK5[indx_four]	            	  = 0.0 + 0.0 * I;
				RK_data->RK6[indx_four]	            	  = 0.0 + 0.0 * I;
				#endif
				#if defined(__DPRK5)
				RK_data->RK7[indx_four]	            	 = 0.0 + 0.0 * I;
				RK_data->psi_hat_last[indx_four]		 = 0.0 + 0.0 * I;
				#endif
			}
			if (i == 0) {
				if (j < Ny_Fourier) {
					run_data->k[1][j] = 0;
				}
				run_data->x[1][j] = 0.0;
			}

		}
		run_data->k[0][i] = 0; 
		run_data->x[0][i] = 0.0;
	}
}
/**
 * Wrapper function that initializes the FFTW plans using MPI
 * @param N Array containing the dimensions of the system
 */
void InitializeFFTWPlans(const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];

	// -----------------------------------
	// Initialize Plans for Vorticity 
	// -----------------------------------
	// Set up FFTW plans for normal transform - vorticity field
	sys_vars->fftw_2d_dft_r2c = fftw_mpi_plan_dft_r2c_2d(Nx, Ny, run_data->psi, run_data->psi_hat, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	sys_vars->fftw_2d_dft_c2r = fftw_mpi_plan_dft_c2r_2d(Nx, Ny, run_data->psi_hat, run_data->psi, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	if (sys_vars->fftw_2d_dft_r2c == NULL || sys_vars->fftw_2d_dft_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}

	// -------------------------------------
	// Initialize batch Plans for Velocity 
	// -------------------------------------
	// Set up FFTW plans for batch transform - velocity fields
	sys_vars->fftw_2d_dft_batch_r2c = fftw_mpi_plan_many_dft_r2c((int)SYS_DIM, N, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u, run_data->u_hat, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	sys_vars->fftw_2d_dft_batch_c2r = fftw_mpi_plan_many_dft_c2r((int)SYS_DIM, N, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u_hat, run_data->u, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	if (sys_vars->fftw_2d_dft_batch_r2c == NULL || sys_vars->fftw_2d_dft_batch_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
}
/**
 * Wrapper function that frees any memory dynamcially allocated in the programme
 * @param RK_data Pointer to a struct contaiing the integraiont arrays
 */
void FreeMemory(RK_data_struct* RK_data) {

	// ------------------------
	// Free memory 
	// ------------------------
	// Free space variables
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}


	// Free system variables
	fftw_free(run_data->psi);
	fftw_free(run_data->psi_hat);
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	#if defined(PHASE_ONLY)
	fftw_free(run_data->a_k);
	fftw_free(run_data->phi_k);
	fftw_free(run_data->tmp_a_k);
	#endif
	#if defined(__NONLIN)
	fftw_free(run_data->nonlinterm);
	#endif
	#if defined(__SYS_MEASURES)
	fftw_free(run_data->tot_energy);
	fftw_free(run_data->tot_enstr);
	fftw_free(run_data->tot_palin);
	fftw_free(run_data->enrg_diss);
	fftw_free(run_data->enst_diss);
	#endif
	#if defined(__ENRG_FLUX)
	fftw_free(run_data->enrg_flux_sbst);
	fftw_free(run_data->enrg_diss_sbst);
	#endif
	#if defined(__ENRG_SPECT)
	fftw_free(run_data->enrg_spect);
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	fftw_free(run_data->enrg_flux_spect);
	#endif
	#if defined(TESTING)
	fftw_free(run_data->exact_soln);
	#endif
	#if defined(__TIME)
	if (!(sys_vars->rank)){
		fftw_free(run_data->time);
	}
	#endif


	// Free integration variables
	fftw_free(RK_data->RK1);
	fftw_free(RK_data->RK2);
	fftw_free(RK_data->RK3);
	fftw_free(RK_data->RK4);
	#if defined(__RK5) || defined(__DPRK5)
	fftw_free(RK_data->RK5);
	fftw_free(RK_data->RK6);
	#endif 
	#if defined(__DPRK5)
	fftw_free(RK_data->RK7);
	fftw_free(RK_data->psi_hat_last);
	#endif
	fftw_free(RK_data->RK_tmp);
	fftw_free(RK_data->grad_psi);
	fftw_free(RK_data->grad_psi_hat);


	// ------------------------
	// Destroy FFTW plans 
	// ------------------------
	fftw_destroy_plan(sys_vars->fftw_2d_dft_r2c);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_c2r);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_r2c);
	fftw_destroy_plan(sys_vars->fftw_2d_dft_batch_c2r);
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------