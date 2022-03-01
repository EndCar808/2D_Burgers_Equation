/**
* @file utils.c  
* @author Enda Carroll
* @date Mar 2022
* @brief File containing the utilities functions for the pseudospectral solver
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h> 
#include <math.h>
#include <complex.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function to read in arguements given at the command line upon execution of the solver
 * @param  argc The number of arguments given
 * @param  argv Array containg the arguments specified
 * @return      Returns 0 if the parsing of arguments has been successful
 */
int GetCMLArgs(int argc, char** argv) {

    // Initialize Variables
    int c;
    int dim_flag        = 0;
    int force_flag      = 0;
    int output_dir_flag = 0;

    // -------------------------------
    // Initialize Default Values
    // -------------------------------
    // Input file path
    strncpy(file_info->input_file_name, "NONE", 512);
    // Output file directory
    strncpy(file_info->output_dir, "../Data/Tmp/", 512);  // Set default output directory to the Tmp folder
    strncpy(file_info->output_tag, "NO_TAG", 64);
    file_info->file_only = 0; // used to indicate if output file should be file only i.e., not output folder
    // System dimensions
    sys_vars->N[0]       = 64;
    sys_vars->N[1]       = 64;
    // Integration time 
    sys_vars->t0         = 0.0;
    sys_vars->dt         = 1e-4;
    sys_vars->T          = 1.0;
    sys_vars->CFL_CONST  = sqrt(3);
    // Initial conditions
    strncpy(sys_vars->u0, "TG_VORT", 64);
    // Forcing
    strncpy(sys_vars->forcing, "NONE", 64); 
    sys_vars->force_k    = 0;
    // System viscosity
    sys_vars->NU         = 1.0;
    // Write to file every 
    sys_vars->SAVE_EVERY = 100;

    // -------------------------------
    // Parse CML Arguments
    // -------------------------------
    while ((c = getopt(argc, argv, "o:h:n:s:e:t:v:i:c:p:f:z:")) != -1) {
        switch(c) {
            case 'o':
                if (output_dir_flag == 0) {
                    // Read in location of output directory
                    strncpy(file_info->output_dir, optarg, 512);
                    output_dir_flag++;
                }
                else if (output_dir_flag == 1) {
                    // Output file only indicated
                    file_info->file_only = 1;
                }
                break;
            case 'n':
                // Read in the dimensions of the system
                sys_vars->N[dim_flag] = atoi(optarg);

                // Check dimensions satisfy requirements
                if (sys_vars->N[dim_flag] <= 2) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The minimum dimension size of [%ld] must be greater than 2\n-->> Exiting!\n\n", sys_vars->N[dim_flag]);       
                    exit(1);
                }
                else if (sys_vars->N[dim_flag] < sys_vars->rank) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The system dimension of [%ld] cannot be less than the number of MPI processes of [%d]\n-->> Exiting!\n\n", sys_vars->N[dim_flag], sys_vars->rank);        
                    exit(1);
                }
                else if (sys_vars->N[dim_flag] % 2 != 0) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The system dimension of [%ld] must be a multiple of 2\n-->> Exiting!\n", sys_vars->N[dim_flag]);      
                    exit(1);
                }
                dim_flag++;
                break;
            case 's':
                // Read in intial time
                sys_vars->t0 = atof(optarg);

                if (sys_vars->t0 < 0) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The integration start time [%lf] must be a positive\n-->> Exiting!\n", sys_vars->t0);      
                    exit(1);
                }   
                break;
            case 'e':
                // Read in final time
                sys_vars->T = atof(optarg); 
                if (sys_vars->T < 0) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The integration end time [%lf] must be a positive\n-->> Exiting!\n", sys_vars->T);      
                    exit(1);
                }
                else if (sys_vars->T < sys_vars->t0) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided end time: [%lf] must be greater than the initial time: [%lf]\n-->> Exiting!\n\n", sys_vars->T, sys_vars->t0);        
                    exit(1);
                }
                break;
            case 'h':
                // Read in initial timestep
                sys_vars->dt = atof(optarg);
                if (sys_vars->dt <= 0) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided timestep: [%lf] must be strictly positive\n-->> Exiting!\n\n", sys_vars->dt);        
                    exit(1);
                }
                break;
            case 'c':
                // Read in value of the CFL -> this can be used to control the timestep
                sys_vars->CFL_CONST = atof(optarg);
                if (sys_vars->CFL_CONST <= 0) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided CFL Constant: [%lf] must be strictly positive\n-->> Exiting!\n\n", sys_vars->CFL_CONST);     
                    exit(1);
                }
                break;
            case 'v':
                // Read in the viscosity
                sys_vars->NU = atof(optarg);
                if (sys_vars->NU <= 0) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The provided viscosity: [%lf] must be strictly positive\n-->> Exiting!\n\n", sys_vars->NU);       
                    exit(1);
                }
                break;
            case 'i':
                // Read in the initial conditions
                if (!(strcmp(optarg,"TG_VORT"))) {
                    // The Taylor Green vortex - starting with the velocity
                    strncpy(sys_vars->u0, "TG_VORT", 64);
                    break;
                }
                else if (!(strcmp(optarg,"TESTING"))) {
                    // Specific ICs for testing - powerlaw amps and phases = pi/4
                    strncpy(sys_vars->u0, "TESTING", 64);
                    break;
                }
                else if (!(strcmp(optarg,"RANDOM"))) {
                    // Random initial conditions
                    strncpy(sys_vars->u0, "RANDOM", 64);
                    break;
                }
                else {
                    // No initial conditions specified -> this will default to random initial conditions
                    strncpy(sys_vars->u0, "NONE", 64);
                    break;
                }
                break;
            case 't':
                // Read in output directory tag
                strncpy(file_info->output_tag, optarg, 64); 
                break;
            case 'p':
                // Read in how often to print to file
                sys_vars->SAVE_EVERY = atoi(optarg);
                break;
            case 'f':
                // Read in the forcing type
                if (!(strcmp(optarg,"ZERO")) && (force_flag == 0)) {
                    // Killing certain modes
                    strncpy(sys_vars->forcing, "ZERO", 64);
                    break;
                }
                else if (!(strcmp(optarg,"KOLM"))  && (force_flag == 0)) {
                    // Kolmogorov forcing
                    strncpy(sys_vars->forcing, "KOLM", 64);
                    break;
                }
                else if ((force_flag == 1)) {
                    // Get the forcing wavenumber
                    sys_vars->force_k = atoi(optarg);
                }
                else {
                    // Set default forcing to None
                    strncpy(sys_vars->forcing, "NONE", 64);
                    break;
                }
                break;
            case 'z':
                strncpy((file_info->input_file_name), optarg, 512); // copy the input file name given as a command line argument
                if ( access((file_info->input_file_name), F_OK) != 0) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] Parsing of Command Line Arguements Failed: The input file [%s] cannot be found, please ensure correct path to file is specified.\n", (file_info->input_file_name));      
                    exit(1);                    
                }
                break;
            default:
                fprintf(stderr, "\n["RED"ERROR"RESET"] Incorrect command line flag encountered\n");     
                fprintf(stderr, "Use"YELLOW" -o"RESET" to specify the output directory\n");
                fprintf(stderr, "Use"YELLOW" -n"RESET" to specify the size of each dimension in the system\n");
                fprintf(stderr, "Use"YELLOW" -s"RESET" to specify the start time of the simulation\n");
                fprintf(stderr, "Use"YELLOW" -e"RESET" to specify the end time of the simulation\n");
                fprintf(stderr, "Use"YELLOW" -h"RESET" to specify the timestep\n");
                fprintf(stderr, "Use"YELLOW" -c"RESET" to specify the CFL constant for the adaptive stepping\n");
                fprintf(stderr, "Use"YELLOW" -v"RESET" to specify the system viscosity\n");
                fprintf(stderr, "Use"YELLOW" -i"RESET" to specify the initial condition\n");
                fprintf(stderr, "Use"YELLOW" -t"RESET" to specify the tag name to be used in the output file directory\n");
                fprintf(stderr, "Use"YELLOW" -f"RESET" to specify the forcing type\n");
                fprintf(stderr, "Use"YELLOW" -p"RESET" to specify how often to print to file\n");
                fprintf(stderr, "Use"YELLOW" -z"RESET" to specify an input file to read parameters from\n");
                fprintf(stderr, "\nExample usage:\n"CYAN"\tmpirun -n 4 ./bin/main -o \"../Data/Tmp\" -n 64 -n 64 -s 0.0 -e 1.0 -h 0.0001 -v 1.0 -i \"TG_VORT\" -t \"TEMP_RUN\" \n"RESET);
                fprintf(stderr, "-->> Now Exiting!\n\n");
                exit(1);
        }
    }

    return 0;
}   
/**
 * Function to print the Real and Fourier space variables
 */
void PrintSpaceVariables(const long int* N) {

    // Initialize variables
    const long int Nx = N[0];
    const long int Ny_Fourier = N[1] / 2 + 1; 

    // Allocate global array memory
    double* x0 = (double* )fftw_malloc(sizeof(double) * Nx);
    int* k0    = (int* )fftw_malloc(sizeof(int) * Nx);

    // Gather the data from each process onto master rank for printing
    MPI_Gather(run_data->x[0], sys_vars->local_Nx, MPI_DOUBLE, x0, sys_vars->local_Nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(run_data->k[0], sys_vars->local_Nx, MPI_INT, k0, sys_vars->local_Nx, MPI_INT, 0, MPI_COMM_WORLD);

    // Print results on the master process
    if ( !(sys_vars->rank) ) {
        for (int i = 0; i < Nx; ++i) {
            if (i < Ny_Fourier) {
                printf("x[%d]: %5.16lf\ty[%d]: %5.16lf\tkx[%d]: %d\tky[%d]: %d\n", i, x0[i], i, run_data->x[1][i], i, k0[i], i, run_data->k[1][i]);
            }
            else {
                printf("x[%d]: %5.16lf\ty[%d]: %5.16lf\tkx[%d]: %d\n", i, x0[i], i, run_data->x[1][i], i, k0[i]);
            }
        }
        printf("\n\n");
    }

    // Free up memory
    fftw_free(x0);
    fftw_free(k0);
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------