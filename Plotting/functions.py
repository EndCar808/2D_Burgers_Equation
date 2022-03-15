#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Mar 2022
## Info: General python functions for analysing
#        Solver data


#######################
##  Library Imports  ##
#######################
import numpy as np
import h5py
import sys
import os
from numba import njit
import pyfftw
from collections.abc import Iterable
from itertools import zip_longest
from subprocess import Popen, PIPE

#################################
## Colour Printing to Terminal ##
#################################
class tc:
    H    = '\033[95m'
    B    = '\033[94m'
    C    = '\033[96m'
    G    = '\033[92m'
    Y    = '\033[93m'
    R    = '\033[91m'
    Rst  = '\033[0m'
    Bold = '\033[1m'
    Underline = '\033[4m'



#################################
##          MISC               ##
#################################
def fft_ishift_freq(w_h, axes = None):

    """
    My version of fft.ifftshift
    """

    ## If no axes provided
    if axes == None:
        ## Create axes tuple
        axes  = tuple(range(w_h.ndim))
        ## Create shift list -> adjusted for FFTW freq numbering
        shift = [-(dim // 2 + 1) for dim in w_h.shape]

    ## If axes is an integer
    elif isinstance(axes, int):
        ## Create the shift object on this axes
        shift = -(w_h.shape[axes] // 2 + 1)

    ## If axes is a tuple
    else:
        ## Create appropriate shift for each axis
        shift = [-(w_h.shape[ax] // 2 + 1) for ax in axes]

    return np.roll(w_h, shift, axes)

def run_commands_parallel(cmd_lsit, proc_limit):

    """
    Runs commands in parallel.

    Input Parameters:
        cmd_list    : list
                     - List of commands to run
        proc_limit  : int
                     - the number of processes to create to execute the commands in parallel
    """

    ## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
    groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmd_list)] * proc_limit

    ## Loop through grouped iterable
    for processes in zip_longest(*groups):
        for proc in filter(None, processes): # filters out 'None' fill values if proc_limit does not divide evenly into cmd_list
            ## Print command to screen
            print("\nExecuting the following command:\n\t" + tc.C + "{}".format(proc.args[0]) + tc.Rst)

            # Communicate with process to retrive output and error
            [run_CodeOutput, run_CodeErr] = proc.communicate()

            ## Print both to screen
            print(run_CodeOutput)
            print(run_CodeErr)

            ## Wait until all finished
            proc.wait()



#####################################
##       DATA FILE FUNCTIONS       ##
#####################################
def sim_data(input_dir, method = "default"):

    """
    Reads in the system parameters from the simulation provided in SimulationDetails.txt.

    Input Parameters:
        input_dir : string
                    - If method == "defualt" is True then this will be the path to
                    the input folder. if not then this will be the input folder
        method    : string
                    - Determines whether the data is to be read in from a file or
                    from an input folder
    """

    ## Define a data class 
    class SimulationData:

        """
        Class for the system parameters.
        """

        ## Initialize class
        def __init__(self, Nx = 0, Ny = 0, Nk = 0, nu = 0.0, t0 = 0.0, T = 0.0, ndata = 0, u0 = "HOPF_COLE", cfl = 0.0, spec_size = 0, dt = 0., dx = 0., dy = 0.):
            self.Nx     = int(Nx)
            self.Ny     = int(Ny)
            self.Nk     = int(Nk)
            self.nu     = float(nu)
            self.t0     = float(t0)
            self.T      = float(T)
            self.ndata  = int(ndata)
            self.u0     = str(u0)
            self.cfl    = float(cfl)
            self.dt     = float(dt)
            self.dx     = float(dx)
            self.dy     = float(dy)
            self.spec_size = int(spec_size)


    ## Create instance of class
    data = SimulationData()

    if method == "default":
        ## Read in simulation data from file
        with open(input_dir + "SimulationDetails.txt") as f:

            ## Loop through lines and parse data
            for line in f.readlines():

                ## Parse viscosity
                if line.startswith('Viscosity'):
                    data.nu = float(line.split()[-1])

                ## Parse number of collocation points
                if line.startswith('Collocation Points'):
                    data.Nx = int(line.split()[-2].lstrip('[').rstrip(','))
                    data.Ny = int(line.split()[-1].split(']')[0])

                ## Parse the number of Fourier modes
                if line.startswith('Fourier Modes'):
                    data.Nk = int(line.split()[-1].split(']')[0])

                ## Parse the start and end time
                if line.startswith('Time Range'):
                    data.t0 = float(line.split()[-3].lstrip('['))
                    data.T  = float(line.split()[-1].rstrip(']'))

                ## Parse number of saving steps
                if line.startswith('Total Saving Steps'):
                    data.ndata = int(line.split()[-1]) + 1 # plus 1 to include initial condition

                ## Parse the initial condition
                if line.startswith('Initial Conditions'):
                    data.u0 = str(line.split()[-1])

                ## Parse the timestep
                if line.startswith('Finishing Timestep'):
                    data.dt = float(line.split()[-1])

                ## Parse the spatial increment
                if line.startswith('Spatial Increment'):
                    data.dy = float(line.split()[-1].rstrip(']'))
                    data.dx = float(line.split()[-2].rstrip(',').lstrip('['))

            ## Get spectrum size
            data.spec_size = int(np.sqrt((data.Nx / 2)**2 + (data.Ny / 2)**2) + 1)
    else:

        for term in input_dir.split('_'):

            ## Parse Viscosity
            if term.startswith("NU"):
                data.nu = float(term.split('[')[-1].rstrip(']'))

            ## Parse Number of collocation points & Fourier modes
            if term.startswith("N["):
                data.Nx = int(term.split('[')[-1].split(',')[0])
                data.Ny = int(term.split('[')[-1].split(',')[-1].rstrip(']'))
                data.Nk = int(data.Nx / 2 + 1)

            ## Parse Time range
            if term.startswith('T['):
                data.t0 = float(term.split('-')[0].lstrip('T['))
                data.T  = float(term.split('-')[-1].rstrip(']'))

            ## Parse CFL number
            if term.startswith('CFL'):
                data.cfl = float(term.split('[')[-1].rstrip(']'))

            ## Parse initial condition
            if term.startswith('u0'):
                data.u0 = str(term.split('[')[-1])
            if not term.startswith('u0') and term.endswith('].h5'):
                data.u0 = data.u0 + '_' + str(term.split(']')[0])

        ## Get the number of data saves
        with h5py.File(input_dir, 'r') as file:
            data.ndata = len([g for g in file.keys() if 'Iter' in g])

        ## Get spectrum size
        data.spec_size = int(np.sqrt((data.Nx / 2)**2 + (data.Ny / 2)**2) + 1)

    return data



def import_data(input_file, sim_data, method = "default"):

    """
    Reads in run data from main HDF5 file.

    input_dir : string
                - If method == "defualt" is True then this will be the path to
               the input folder. if not then this will be the input folder
    method    : string
                - Determines whether the data is to be read in from a file or
               from an input folder
    sim_data  : class
                - object containing the simulation parameters
    """


    ## Define a data class for the solver data
    class SolverData:

        """
        Class for the run data.
        """

        def __init__(self):
            ## Allocate global arrays
            self.psi        = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny))
            self.u          = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny, 2))
            self.exact_soln = np.zeros((sim_data.ndata, sim_data.Nx, sim_data.Ny))
            self.u_hat      = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk, 2)) * np.complex(0.0, 0.0)
            self.psi_hat    = np.ones((sim_data.ndata, sim_data.Nx, sim_data.Nk)) * np.complex(0.0, 0.0)
            self.time       = np.zeros((sim_data.ndata, ))
            ## Allocate system measure arrays
            self.tot_enrg  = np.zeros((int(sim_data.ndata * 2), ))
            self.enrg_diss = np.zeros((int(sim_data.ndata * 2), ))
            self.enrg_diss_sbst = np.zeros((int(sim_data.ndata * 2), ))
            self.enrg_flux_sbst = np.zeros((int(sim_data.ndata * 2), ))
            ## Allocate spatial arrays
            self.kx    = np.zeros((sim_data.Nx, ))
            self.ky    = np.zeros((sim_data.Nk, ))
            self.x     = np.zeros((sim_data.Nx, ))
            self.y     = np.zeros((sim_data.Ny, ))
            self.k2    = np.zeros((sim_data.Nx, sim_data.Nk))
            self.k2Inv = np.zeros((sim_data.Nx, sim_data.Nk))

    ## Create instance of data class
    data = SolverData()

    ## Depending on the output mmode of the solver the input files will be named differently
    if method == "default":
        in_file = input_file + "Main_HDF_Data.h5"
    else:
        in_file = input_file

    ## Open file and read in the data
    with h5py.File(in_file, 'r') as file:

        ## Initialize counter
        nn = 0

        # Read in the vorticity
        for group in file.keys():
            if "Iter" in group:
                if 'psi' in list(file[group].keys()):
                    data.psi[nn, :, :] = file[group]["psi"][:, :]
                if 'psi_hat' in list(file[group].keys()):
                    data.psi_hat[nn, :, :] = file[group]["psi_hat"][:, :]
                if 'u' in list(file[group].keys()):
                    data.u[nn, :, :] = file[group]["u"][:, :, :]
                if 'u_hat' in list(file[group].keys()):
                    data.u_hat[nn, :, :] = file[group]["u_hat"][:, :, :]
                if 'ExactSoln' in list(file[group].keys()):
                    data.exact_soln[nn, :, :] = file[group]["ExactSoln"][:, :]
                data.time[nn] = file[group].attrs["TimeValue"]
                nn += 1
            else:
                continue

        ## Read in the space arrays
        if 'kx' in list(file.keys()):
            data.kx = file["kx"][:]
        if 'ky' in list(file.keys()):
            data.ky = file["ky"][:]
        if 'x' in list(file.keys()):
            data.x  = file["x"][:]
        if 'y' in list(file.keys()):
            data.y  = file["y"][:]

        ## Read system measures
        if 'TotalEnergy' in list(file.keys()):
            data.tot_enrg = file['TotalEnergy'][:]
        if 'EnergyDissipation' in list(file.keys()):
            data.enrg_diss = file['EnergyDissipation'][:]
        if 'EnergyDissSubset' in list(file.keys()):
            data.enrg_diss_sbst = file['EnergyDissSubset'][:]
        if 'EnergyFluxSubset' in list(file.keys()):
            data.enrg_flux_sbst = file['EnergyFluxSubset'][:]

    ## Get inv wavenumbers
    data.k2 = data.ky**2 + data.kx[:, np.newaxis]**2
    index   = data.k2 != 0.0
    data.k2Inv[index] = 1. / data.k2[index]

    return data


