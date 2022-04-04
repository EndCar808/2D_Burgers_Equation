#!/usr/bin/env python3    

## Author: Enda Carroll
## Date: Mar 2022
## Info: Script to run test of solver data

#######################
##  LIBRARY IMPORTS  ##
#######################
import numpy as np
import pyfftw as fftw
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import getopt
from subprocess import Popen, PIPE
from itertools import zip_longest
from Plotting.functions import tc

########################
## 	 FUNCTION DEFS    ##
########################
def get_cmd_args(argv):

	"""
	Parses command line arguments
	"""

	## Create arguments class
	class cmd_args:

		"""
		Class for command line arguments
		"""

		def __init__(self, out_dir = "./Data/Test", mode = None):
			self.test_mode = mode
			self.out_dir   = out_dir


	## Initialize class
	cargs = cmd_args()

	try:
		## Gather command line arguments
		opts, args = getopt.getopt(argv, "i:o:m:s:f:", ["mode=", "plot"])
	except:
		print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
		raise

	## Parse command line args
	for opt, arg in opts:
		if opt in ['--mode']:
			## Read in test_mode
			cargs.test_mode = str(arg)

	return cargs

def run_code(cmd_list, num_procs):

	## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
	groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmd_list)] * num_procs

	## Loop through grouped iterable
	for processes in zip_longest(*groups):
		for proc in filter(None, processes): # filters out 'None' fill values if proc_limit does not divide evenly into cmd_list
		    ## Print command to screen
		    print("\nExecuting the following command:\n\n\t" + tc.C + "{}\n".format(proc.args[:]) + tc.Rst)

		    ## Communicate with process to retrive output and error
		    [run_CodeOutput, run_CodeErr] = proc.communicate()

		    ## Print both to screen
		    print(run_CodeOutput)
		    print(run_CodeErr)

		    ## Wait until all finished
		    proc.wait()

def run_c_code(executable, num_procs, Nx, Ny, t0, T, dt, tag, u0, nu, output_dir):

	## Check if C code executable exists and run it to gather test data
	if not os.path.isfile(executable):
		print("[" + tc.R + "ERROR" + tc.Rst + "] ---> C Code Executable at [" + tc.C + "{}" + tc.Rst + "] missing.".format(executable))
		sys.exit()
	else:
		##-------- Run C code solver
		## Generate command
		c_cmd = ["mpirun -n {} {} -n {} -n {} -s {} -e {} -h {} -c 0.900000 -t {} -i {} -v {} -o {} -o 1 -f 0 -p 1".format(num_procs, executable, Nx, Ny, t0, T, dt, tag, u0, nu, output_dir)]

		## Run command
		run_code(c_cmd, 1)

def run_py_code(executable, Nx, Ny, t0, T, dt, tag, u0, nu, output_dir):

	## Check if Python code exists and run it to gather test data
	if not os.path.isfile(executable):
		print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Python code at [" + tc.C + "{}" + tc.Rst + "] missing.".format(executable))
		sys.exit()
	else:
		##-------- Run Python Code solver
		## Generate command
		py_cmd = ["python3 {} -n {} -s {} -e {} -h {} -t {} -u {} -v {} -p {} -o {}".format(executable, Nx, t0, T, dt, tag, u0, nu, 1, output_dir)]

		## Run command
		run_code(py_cmd, 1)

########################
## 	 	 MAIN   	  ##
########################
if __name__ == "__main__":

	##################################
	##       GET TEST VARIABLES     ##
	##################################
	## Parse command line arguments
	cmdargs = get_cmd_args(sys.argv[1:])

	if cmdargs.test_mode in ["SOLVER"]:
		
		## Get test case parameters
		Nx        = 16
		Ny        = 16
		nu        = 1e-3
		t0        = 0.0
		T         = 1.0
		dt        = 1e-3
		u0        = "COLE_HOPF"
		tag       = "COLE-HOPF-TEST"
		num_procs = 4
		output_dir = cmdargs.out_dir + "/SOLVER_TEST_N[{},{}]_T[{}-{}]_NU[{:0.6f}]_u0[{}]_TAG[{}]/".format(Nx, Ny, t0, T, nu, u0, tag)

	## Code excutables
	c_code_exec  = "/home/ecarroll/PhD/2D_Burgers_Equation/Solver/bin/solver_test"
	py_code_exec = "/home/ecarroll/PhD/2D_Burgers_Equation/TestingFunctions/solver.py"

	##############################
	##        OUTPUT FOLDER     ##
	##############################
	## Check if test data folder exists
	if not os.path.isdir(output_dir):
		print(tc.Y + "Making output folder..." + tc.Rst)
		os.mkdir(output_dir)
		print("Output folder:" + tc.C + " {}".format(output_dir) + tc.Rst)

		################################
		##     GENERATE TEST DATA     ##
		################################
		print(tc.Y + "Running Solvers..." + tc.Rst)
		## Run c code
		run_c_code(c_code_exec, num_procs, Nx, Ny, t0, T, dt, tag, u0, nu, output_dir)

		## Run python code
		run_py_code(py_code_exec, Nx, Ny, t0, T, dt, tag, u0, nu, output_dir)
	elif os.path.isdir(output_dir) and os.listdir(output_dir) == []:
		################################
		##     GENERATE TEST DATA     ##
		################################
		print(tc.Y + "Running Solvers..." + tc.Rst)
		## Run c code
		run_c_code(c_code_exec, num_procs, Nx, Ny, t0, T, dt, tag, u0, nu, output_dir)

		## Run python code
		run_py_code(py_code_exec, Nx, Ny, t0, T, dt, tag, u0, nu, output_dir)
	

	if os.path.isdir(output_dir):
		##############################
		##     GATHER TEST DATA     ##
		##############################
		## Check if data exists
		print(tc.Y + "Gathering data..." + tc.Rst)
		c_code_data      = ""
		c_code_test_data = ""
		py_code_data     = ""
		for f in os.listdir(output_dir):

			## Check if C code data is there if not generate it
			if "Main" in f:
				c_code_data = output_dir + f

				with h5py.File(c_code_data, 'r') as c_file:
					c_x       = c_file["x"][:]
					c_y       = c_file["y"][:]
			## Check if C code test data is there if not generate it
			if "Test_Data" in f:
				c_code_test_data = output_dir + f

				with h5py.File(c_code_test_data, 'r') as ct_file:
					## Get the number of snapshots
					ndata = len([g for g in ct_file.keys() if 'Iter' in g])

					## Get the space arrays
					ct_x = ct_file["x"][:]
					ct_y = ct_file["y"][:]

					## Read in the snapshot data
					nn = 0
					ct_psi     = np.zeros((ndata, Nx, Ny))
					ct_psi_hat = np.ones((ndata, Nx, Ny//2 + 1)) * np.complex(0.0, 0.0)
					ct_RK1     = np.ones((ndata, Nx, Ny//2 + 1)) * np.complex(0.0, 0.0)
					for group in ct_file.keys():
						if "Iter" in group:
							if 'psi' in list(ct_file[group].keys()):
								ct_psi[nn, :, :] = ct_file[group]["psi"][:, :]
							if 'psi_hat' in list(ct_file[group].keys()):
								ct_psi_hat[nn, :, :] = ct_file[group]["psi_hat"][:, :]
							if 'RK1' in list(ct_file[group].keys()):
								ct_RK1[nn, :, :] = ct_file[group]["RK1"][:, :]
							nn += 1
							
			## Check if python code data exists if not create it
			if "PyTestData" in f:
				py_code_data = output_dir + f

				## Read in data
				with h5py.File(py_code_data, 'r') as py_file:
					## Get the number of snapshots
					ndata = len([g for g in py_file.keys() if 'Iter' in g])

					## Get the space arrays
					py_x = py_file["x"][:]
					py_y = py_file["y"][:]

					## Read in the snapshot data
					nn = 0
					py_psi     = np.zeros((ndata, Nx, Ny))
					py_psi_hat = np.ones((ndata, Nx, Ny//2 + 1)) * np.complex(0.0, 0.0)
					py_RK1     = np.ones((ndata, Nx, Ny//2 + 1)) * np.complex(0.0, 0.0)
					for group in py_file.keys():
						if "Iter" in group:
							if 'psi' in list(py_file[group].keys()):
								py_psi[nn, :, :] = py_file[group]["psi"][:, :]
							if 'psi_hat' in list(py_file[group].keys()):
								py_psi_hat[nn, :, :] = py_file[group]["psi_hat"][:, :]
							if 'RK1' in list(py_file[group].keys()):
								py_RK1[nn, :, :] = py_file[group]["RK1"][:, :]
							nn += 1
		

		## If there was data files missing create them now
		if not c_code_data or not c_code_test_data:
			print(tc.Y + "No C code data...now creating" + tc.Rst)
			## Run c code if code file not there
			run_c_code(c_code_exec, num_procs, Nx, Ny, t0, T, dt, tag, u0, nu, output_dir)

		if not py_code_data:
			print(tc.Y + "No Python code data...now creating" + tc.Rst)
			## Run python code file if python code not there
			run_py_code(py_code_exec, Nx, Ny, t0, T, dt, tag, u0, nu, output_dir)


		###############################
		##     COMPARE TEST DATA     ##
		###############################
		abs_tol = 1e-08  ## 1e-08
		rel_tol = 1e-05  ## 1e-05
		print("Compare Space Data")
		print(py_x.shape == ct_x.shape)
		print(py_y.shape == ct_y.shape)
		print(np.allclose(ct_x, py_x, rtol = rel_tol, atol = abs_tol))
		print(np.allclose(ct_y, py_y, rtol = rel_tol, atol = abs_tol))

		print("Compare Shape")
		print(py_psi.shape == ct_psi.shape)
		print(py_psi_hat.shape == ct_psi_hat.shape)

		print("Compare Psi Data")
		print(np.linalg.norm(py_psi[0, :, :] - ct_psi[0, :, :]))
		print(np.linalg.norm(py_psi[0, :, :] - ct_psi[0, :, :], ord = np.inf))

		print("Compare Psi Hat Data")
		print(np.linalg.norm(py_psi_hat[0, :, :] - ct_psi_hat[0, :, :]))
		print(np.linalg.norm(py_psi_hat[0, :, :] - ct_psi_hat[0, :, :], ord = np.inf))

		print("Compare Psi / Psi_hat Data")
		for t in range(np.amax([py_RK1.shape[0], ct_RK1.shape[0]])):
			print("t = {} - L2   | psi_hat: {:0.10g} \t - psi: {:0.10g}".format(t, np.linalg.norm(py_psi_hat[t, :, :] - ct_psi_hat[t, :, :]), np.linalg.norm(py_psi[t, :, :] - ct_psi[t, :, :])))
			print("t = {} - Linf | psi_hat: {:0.10g} \t - psi: {:0.10g}".format(t, np.linalg.norm(py_psi_hat[t, :, :] - ct_psi_hat[t, :, :], ord = np.inf), np.linalg.norm(py_psi[t, :, :] - ct_psi[t, :, :], ord = np.inf)))