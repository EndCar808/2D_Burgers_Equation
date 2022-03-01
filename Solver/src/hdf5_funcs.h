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
hid_t CreateComplexDatatype(void);
hid_t CreateGroup(hid_t file_handle, char* filename, char* group_name, double t, double dt, long int iters);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------