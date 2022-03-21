/**
* @file hdf5_funcs.h  
* @author Enda Carroll
* @date Mar 2022
* @brief File containing function prototpyes for hdf5_funcs file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
//  Function Prototpyes
// ---------------------------------------------------------------------
void CreateOutputFilesWriteICs(const long int* N, double dt);
void GetOutputDirPath(void);
void OpenTestingFile(void);
void WriteTestDataFourier(fftw_complex* data, char* dset_name, int dim_x, int dim_y, int local_dim_x);
void WriteTestDataReal(double* data, char* dset_name, int dset_rank, int* dset_dims, int local_dim_x);
hid_t CreateGroup(hid_t file_handle, char* filename, char* group_name, double t, double dt, long int iters);
void WriteDataToFile(double t, double dt, long int iters);
void WriteDataFourier(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, fftw_complex* data);
void WriteDataReal(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, double* data);
void WriteDataSpect(double t, int iters, hid_t group_id, int dims, char* dset_name, double* data);
void FinalWriteAndCloseOutputFile(const long int* N, int iters, int save_data_indx);
hid_t CreateComplexDatatype(void);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------