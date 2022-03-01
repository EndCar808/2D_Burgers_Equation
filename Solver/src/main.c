/**
* @file main.c 
* @author Enda Carroll
* @date Mar 2022
* @brief Main file for calling the pseudospectral solver on the 2D Burgers Equation
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <gsl/gsl_histogram.h> 
#include <gsl/gsl_statistics.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_funcs.h"
#include "solver.h"
#include "utils.h"
// ---------------------------------------------------------------------
//  Global Variables Declarations
// ---------------------------------------------------------------------
// Define the global points that will be pointed to the global structs
runtime_data_struct*      run_data;
system_vars_struct*       sys_vars;
HDF_file_info_struct*    file_info;

// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// Create instances of global variables structs
	runtime_data_struct runtime_data;
	system_vars_struct   system_vars;
	HDF_file_info_struct   HDF_file_info;
	
	// Point the global pointers to these structs
	run_data  = &runtime_data;
	sys_vars  = &system_vars;
	file_info = &HDF_file_info;

	//////////////////////////////////
	// Initialize MPI section
	MPI_Init(&argc, &argv);
	//////////////////////////////////
	
	// Get the number of active processes and their rank and print to screen
	MPI_Comm_size(MPI_COMM_WORLD, &(sys_vars->num_procs));      
	MPI_Comm_rank(MPI_COMM_WORLD, &(sys_vars->rank));  
	if ( !(sys_vars->rank) ) {
		printf("\nTotal number of MPI tasks running: %d\n\n", sys_vars->num_procs);
	}

	// Initialize FFTW MPI interface - must be called after MPI_Init but before anything else in FFTW
	fftw_mpi_init();

	// Start timer
	MPI_Barrier(MPI_COMM_WORLD);
	clock_t begin = clock();
	
	// Read in Command Line Arguments
	if (GetCMLArgs(argc, argv) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line aguments, check utils.c file for details\n");
		exit(1);
	}

	//////////////////////////////////
	// Call Solver
	//////////////////////////////////
	SpectralSolve();
	//////////////////////////////////
	// Call Solver
	//////////////////////////////////
	

	MPI_Barrier(MPI_COMM_WORLD);
	if (!(sys_vars->rank)) {
		// Finish timing
		clock_t end = clock();

		// calculate execution time
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		int hh = (int) time_spent / 3600;
		int mm = ((int )time_spent - hh * 3600) / 60;
		int ss = time_spent - hh * 3600 - mm * 60;
		printf("\n\nTotal Execution Time: ["CYAN"%5.10lf"RESET"] --> "CYAN"%d"RESET" hrs : "CYAN"%d"RESET" mins : "CYAN"%d"RESET" secs\n\n", time_spent, hh, mm, ss);

		// Print simulation details to .txt file in default output mode
		if (!file_info->file_only) {
			PrintSimulationDetails(argc, argv, time_spent);
		}
	}

	// Cleanup FFTW MPI interface - Calls the serial fftw_cleanup function also
	fftw_mpi_cleanup();    

	//////////////////////////////////
	// Exit MPI scetion
	MPI_Finalize();
	//////////////////////////////////
	



	// Return statement
	return 0;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
