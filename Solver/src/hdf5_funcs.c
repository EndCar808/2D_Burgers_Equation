/**
* @file hdf5_funcs.c  
* @author Enda Carroll
* @date Mar 2022
* @brief File containing HDF5 function wrappers for creating, opening, wrtining to and closing output file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "solver.h"
#include "data_types.h"
#include "hdf5_funcs.h"
#include "utils.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Wrapper function that creates the ouput directory creates and opens the main output file using parallel access and the spectra file using normal serial access
 */
void CreateOutputFilesWriteICs(const long int* N, double dt) {

    // Initialize variabeles
    const long int Nx         = N[0];
    const long int Ny         = N[1];
    const long int Ny_Fourier = N[1] / 2 + 1;
    hid_t main_group_id;
    #if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
    hid_t spectra_group_id;
    #endif
    char group_name[128];
    herr_t status;
    hid_t plist_id;
    int tmp;
    int indx;

    #if (defined(__VORT_FOUR) || defined(__MODES)) && !defined(DEBUG)
    // Create compound datatype for the complex datasets
    file_info->COMPLEX_DTYPE = CreateComplexDatatype();
    #endif
        
    ///////////////////////////
    /// Create & Open Files
    ///////////////////////////
    // -----------------------------------
    // Create Output Directory and Path
    // -----------------------------------
    GetOutputDirPath();

    // ------------------------------------------
    // Create Parallel File PList for Main File
    // ------------------------------------------
    // Create proptery list for main file access and set to parallel I/O
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    status   = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    if (status < 0) {
        printf("\n["RED"ERROR"RESET"] --- Could not set parallel I/O access for HDF5 output file! \n-->>Exiting....\n");
        exit(1);
    }

    // ---------------------------------
    // Create the output files
    // ---------------------------------
    // Create the main output file
    file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    if (file_info->output_file_handle < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create main HDF5 output file at: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->output_file_name);
        exit(1);
    }

    #if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
    if (!sys_vars->rank){
        // Create the spectra output file
        file_info->spectra_file_handle = H5Fcreate(file_info->spectra_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_info->output_file_handle < 0) {
            fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create HDF5 spectra output file at: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->spectra_file_name);
            exit(1);
        }
    }
    #endif

    ////////////////////////////////
    /// Write Initial Condtions
    ////////////////////////////////
    // --------------------------------------
    // Create Group for Initial Conditions
    // --------------------------------------
    // Initialize Group Name
    sprintf(group_name, "/Iter_%05d", 0);
    
    // Create group for the current iteration data
    main_group_id = CreateGroup(file_info->output_file_handle, file_info->output_file_name, group_name, 0.0, dt, 0);
    #if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
    if (!sys_vars->rank) {
        spectra_group_id = CreateGroup(file_info->spectra_file_handle, file_info->spectra_file_name, group_name, 0.0, dt, 0);
    }
    #endif
}
/**
 * Function that creates the output file paths and directories
 */
void GetOutputDirPath(void) {

    // Initialize variables
    char sys_type[64];
    char solv_type[64];
    char model_type[64];
    char tmp_path[512];
    char file_data[512];  
    struct stat st = {0};   // this is used to check whether the output directories exist or not.

    // ----------------------------------
    // Check if Provided Directory Exists
    // ----------------------------------
    if (!sys_vars->rank) {
        // Check if output directory exists
        if (stat(file_info->output_dir, &st) == -1) {
            printf("\n["YELLOW"NOTE"RESET"] --- Provided Output directory doesn't exist, now creating it...\n");
            // If not then create it
            if ((mkdir(file_info->output_dir, 0700)) == -1) {
                fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create provided output directory ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->output_dir);
                exit(1);
            }
        }
    }

    ////////////////////////////////////////////
    // Check if Output File Only is Requested
    ////////////////////////////////////////////
    if (file_info->file_only) {
        // Update to screen that file only output option is selected
        printf("\n["YELLOW"NOTE"RESET"] --- File only output option selected...\n");
        
        // ----------------------------------
        // Get Simulation Details
        // ----------------------------------
        #if defined(__VISCOUS)
        sprintf(sys_type, "%s", "VIS");
        #elif defined(__INVISCID)
        sprintf(sys_type, "%s", "INVIS");
        #else
        sprintf(sys_type, "%s", "UKN");
        #endif
        #if defined(__RK4)
        sprintf(solv_type, "%s", "RK4");
        #elif defined(__RK5)
        sprintf(solv_type, "%s", "RK5");
        #elif defined(__DPRK5)
        sprintf(solv_type, "%s", "DP5");
        #else 
        sprintf(solv_type, "%s", "UKN");
        #endif
        #if defined(__PHASE_ONLY)
        sprintf(model_type, "%s", "PO");
        #else
        sprintf(model_type, "%s", "FULL");
        #endif

        // -------------------------------------
        // Get File Label from Simulation Data
        // -------------------------------------
        // Construct file label from simulation data
        sprintf(file_data, "_SIM[%s-%s-%s]_N[%ld,%ld]_T[%d-%d]_NU[%1.6lf]_CFL[%1.2lf]_u0[%s].h5", sys_type, solv_type, model_type, sys_vars->N[0], sys_vars->N[1], (int )sys_vars->t0, (int )sys_vars->T, sys_vars->NU, sys_vars->CFL_CONST, sys_vars->u0);

        // ----------------------------------
        // Construct File Paths
        // ---------------------------------- 
        // Construct main file path
        strcpy(tmp_path, file_info->output_dir);
        strcat(tmp_path, "Main_HDF_Data"); 
        strcpy(file_info->output_file_name, tmp_path); 
        strcat(file_info->output_file_name, file_data);
        if ( !(sys_vars->rank) ) {
            printf("\nMain Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);
        }

        #if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
        if ( !(sys_vars->rank) ) {
            // Construct Spectra file path
            strcpy(tmp_path, file_info->output_dir);
            strcat(tmp_path, "Spectra_HDF_Data"); 
            strcpy(file_info->spectra_file_name, tmp_path); 
            strcat(file_info->spectra_file_name, file_data);
            printf("Spectra Output File: "CYAN"%s"RESET"\n\n", file_info->spectra_file_name);
        }   
        #endif
    }
    else {
        // ----------------------------------
        // Get Simulation Details
        // ----------------------------------
        #if defined(__VISCOUS)
        sprintf(sys_type, "%s", "VIS");
        #elif defined(__INVISCID)
        sprintf(sys_type, "%s", "INVIS");
        #else
        sprintf(sys_type, "%s", "SYS_UNKN");
        #endif
        #if defined(__RK4)
        sprintf(solv_type, "%s", "RK4");
        #elif defined(__RK5)
        sprintf(solv_type, "%s", "RK5");
        #elif defined(__DPRK5)
        sprintf(solv_type, "%s", "DP5");
        #else 
        sprintf(solv_type, "%s", "SOLV_UKN");
        #endif
        #if defined(__PHASE_ONLY)
        sprintf(model_type, "%s", "PHAEONLY");
        #else
        sprintf(model_type, "%s", "FULL");
        #endif

        // ----------------------------------
        // Construct Output folder
        // ----------------------------------
        // Construct file label from simulation data
        sprintf(file_data, "SIM_DATA_%s_%s_%s_N[%ld,%ld]_T[%d-%d]_NU[%1.6lf]_CFL[%1.2lf]_u0[%s]_TAG[%s]/", sys_type, solv_type, model_type, sys_vars->N[0], sys_vars->N[1], (int )sys_vars->t0, (int )sys_vars->T, sys_vars->NU, sys_vars->CFL_CONST, sys_vars->u0, file_info->output_tag);

        // ----------------------------------
        // Check Existence of Output Folder
        // ----------------------------------
        strcat(file_info->output_dir, file_data);
        if (!sys_vars->rank) {
            // Check if folder exists
            if (stat(file_info->output_dir, &st) == -1) {
                // If not create it
                if ((mkdir(file_info->output_dir, 0700)) == -1) {
                    fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create folder for output files ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->output_dir);
                    exit(1);
                }
            }
        }

        // ----------------------------------
        // Construct File Paths
        // ---------------------------------- 
        // Construct main file path
        strcpy(file_info->output_file_name, file_info->output_dir); 
        strcat(file_info->output_file_name, "Main_HDF_Data.h5");
        if ( !(sys_vars->rank) ) {
            printf("\nMain Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);
        }

        #if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
        if ( !(sys_vars->rank) ) {
            // Construct spectra file path
            strcpy(file_info->spectra_file_name, file_info->output_dir); 
            strcat(file_info->spectra_file_name, "Spectra_HDF_Data.h5");
            printf("Spectra Output File: "CYAN"%s"RESET"\n\n", file_info->spectra_file_name);
        }   
        #endif
    }

    // Make All process wait before opening output files later
    MPI_Barrier(MPI_COMM_WORLD);
}
/**
 * Wrapper function used to create a Group for the current iteration in the HDF5 file 
 * @param  group_name The name of the group - will be the Iteration counter
 * @param  t          The current time in the simulation
 * @param  dt         The current timestep being used
 * @param  iters      The current iteration counter
 * @return            Returns a hid_t identifier for the created group
 */
hid_t CreateGroup(hid_t file_handle, char* filename, char* group_name, double t, double dt, long int iters) {

    // Initialize variables
    herr_t status;
    hid_t attr_id;
    hid_t group_id;
    hid_t attr_space;
    static const hsize_t attrank = 1;
    hsize_t attr_dims[attrank];

    // -------------------------------
    // Create the group
    // -------------------------------
    // Check if group exists
    if (H5Lexists(file_handle, group_name, H5P_DEFAULT)) {      
        // Open group if it already exists
        group_id = H5Gopen(file_handle, group_name, H5P_DEFAULT);
    }
    else {
        // If not create new group and add time data as attribute to Group
        group_id = H5Gcreate(file_handle, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);   

        // -------------------------------
        // Write Timedata as Attribute
        // -------------------------------
        // Create attribute datatspace
        attr_dims[0] = 1;
        attr_space   = H5Screate_simple(attrank, attr_dims, NULL);  

        // Create attribute for current time in the integration
        attr_id = H5Acreate(group_id, "TimeValue", H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
        if ((H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &t)) < 0) {
            fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not write current time as attribute to group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
            exit(1);
        }
        status = H5Aclose(attr_id);
        if (status < 0 ) {
            fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
            exit(1);
        }
        // Create attribute for the current timestep
        attr_id = H5Acreate(group_id, "TimeStep", H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
        if ((H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &dt)) < 0) {
            fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not write current timestep as attribute to group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
            exit(1);
        }


        // -------------------------------
        // Close the attribute identifiers
        // -------------------------------
        status = H5Aclose(attr_id);
        if (status < 0 ) {
            fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
            exit(1);
        }
        status = H5Sclose(attr_space);
        if (status < 0 ) {
            fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
            exit(1);
        }
    }

    return group_id;
}
/**
 * Function to create a HDF5 datatype for complex data
 */
hid_t CreateComplexDatatype(void) {

    // Declare HDF5 datatype variable
    hid_t dtype;

    // error handling var
    herr_t status;
    
    // Create complex struct
    struct complex_type_tmp cmplex;
    cmplex.re = 0.0;
    cmplex.im = 0.0;

    // create complex compound datatype
    dtype  = H5Tcreate(H5T_COMPOUND, sizeof(cmplex));

    // Insert the real part of the datatype
    status = H5Tinsert(dtype, "r", offsetof(complex_type_tmp,re), H5T_NATIVE_DOUBLE);
    if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert real part for the Complex Compound Datatype!!\nExiting...\n");
        exit(1);
    }

    // Insert the imaginary part of the datatype
    status = H5Tinsert(dtype, "i", offsetof(complex_type_tmp,im), H5T_NATIVE_DOUBLE);
    if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert imaginary part for the Complex Compound Datatype! \n-->>Exiting...\n");
        exit(1);
    }

    return dtype;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------