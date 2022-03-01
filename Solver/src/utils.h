/**
* @file utils.h
* @author Enda Carroll
* @date Mar 2022
* @brief Header file for the utils.c file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  Function Prototypes
// ---------------------------------------------------------------------
// Command  Line Arguments
int GetCMLArgs(int argc, char** argv);
void PrintSimulationDetails(int argc, char** argv, double sim_time);
void PrintSpaceVariables(const long int* N);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------