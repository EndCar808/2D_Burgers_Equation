#!/usr/bin/env python   

## Author: Enda Carroll
## Date: April 2022
## Info: For quickly view solver output data

#######################
##  LIBRARY IMPORTS  ##
#######################
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import os
import sys
import h5py
import shutil
import signal
import matplotlib.pyplot as plt
from itertools import zip_longest
from mpl_toolkits.axes_grid1 import make_axes_locatable
from subprocess import Popen, PIPE, run
from Plotting.functions import tc

############################
##  FUNCTION DEFINITIONS  ##
############################
def run_command_live(cmd):

	'''
	Runs the provided list of commands in the terminal in parallel
	'''

	print("Executing the following command:\n\t" + tc.C + "{}".format(cmd[0]) + tc.Rst)

	## Run cmd in terminal 
	proc = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True)


	## Print output to terminal as it comes
	for line in proc.stdout:
		sys.stdout.write(line)

	# Communicate with process to retrive output and error
	[run_CodeOutput, run_CodeErr] = proc.communicate()

	## Print code error if any
	print(run_CodeErr)

	proc.wait()

def exit_handler(signum, frame):

	'''
	Emtpies Live data directory before exiting programme
	'''

	## Clean live directory and exit
	run(["rm -r {}".format(output_dir + "*")], shell = True)
	print()
	sys.exit(0)

######################
##       MAIN       ##
######################
if __name__ == '__main__':

	## Set upt the signal handler to call our exit function when 'Ctrl+C' is executed at CML
	signal.signal(signal.SIGINT, exit_handler)

	##########################
	##   SOLVER VARIABLES   ##
	##########################
	## Set default 
	solver_procs   = 4
	executable     = "./Solver/bin/solver"
	output_dir     = "Data/Live/"
	Nx             = 128
	Ny             = 128
	t0             = 0.0
	h              = 1e-4
	T              = 50 * h
	c              = 0.7
	v              = 1e-5
	hypervisc      = int(1)
	ekmn_alpha     = 0.0
	ekmn_hypo_diff = -int(2)
	u0             = "HOPF_COLE"
	s_tag          = "Live"
	forcing 	   = 0
	force_k        = 0
	ndata          = 10
	save_every     = 1 ## int((T - t0) / (h * ndata))

	## Clean output directory
	run(["rm -r {}".format(output_dir + "*")], shell = True)

	#########################
	##      RUN SOLVER     ##
	#########################
	## Generate command list 
	cmd_list = ["mpirun -n {} {} -o {} -n {} -n {} -s {:3.1f} -e {:3.6f} -c {:1.6f} -h {:1.6f} -v {:1.10f} -v {} -a {:1.6f} -a {} -i {} -t {} -f {} -f {} -p {}".format(
				solver_procs, 
				executable, 
				output_dir, 
				Nx, Ny, 
				t0, T, 
				c, h, 
				v, hypervisc,
				ekmn_alpha, int(ekmn_hypo_diff), 
				u0, 
				s_tag, 
				forcing, force_k, 
				save_every)]

	## Run command
	run_command_live(cmd_list)

	###########################
	##      DISPLAY DATA     ##
	###########################
	## Read in data for plotting
	data_file = output_dir + os.listdir(output_dir)[0]
	with h5py.File(data_file + "/Main_HDF_Data.h5", 'r') as data:
		print("\nData File: " + tc.C + "{}".format(data_file + "/Main_HDF_Data.h5") + tc.Rst)
		
		## Get the number of snapshots
		ndata = len([g for g in data.keys() if 'Iter' in g])

		print("\nNumber of snapshots: " + tc.C + "{}\n".format(ndata) + tc.Rst)

		## Get the data
		psi = np.zeros((ndata, Nx, Ny))

		nn = 0

		# Read in the data
		for group in data.keys():
			if "Iter" in group:
				if 'psi' in list(data[group].keys()):
					psi[nn, :, :] = data[group]["psi"][:, :]
				nn += 1
		if 'Time' in list(data.keys()):
			time = data['Time'][:]
		if 'TotalEnergy' in list(data.keys()):
			tot_enrg = data['TotalEnergy'][:]
		if 'TotalDivergenceSquared' in list(data.keys()):
			tot_div_sqr = data['TotalDivergenceSquared'][:]
		if 'Totaluv' in list(data.keys()):
			tot_uv = data['Totaluv'][:]
		if 'TotaluSqrvSqr' in list(data.keys()):
			tot_u_sqr_v_sqr = data['TotaluSqrvSqr'][:]

	with h5py.File(data_file + "/Spectra_HDF_Data.h5", 'r') as data:
		print("\nSpectra File: " + tc.C + "{}".format(data_file + "/Spectra_HDF_Data.h5") + tc.Rst)
		
		## Get the number of snapshots
		ndata = len([g for g in data.keys() if 'Iter' in g])

		print("\nNumber of snapshots: " + tc.C + "{}\n".format(ndata) + tc.Rst)

		## Get the data
		enrg_spec = np.zeros((ndata, int(np.sqrt((Nx/2)**2 + (Ny/2)**2) + 1)))

		nn = 0

		# Read in the data
		for group in data.keys():
			if "Iter" in group:
				if 'EnergySpectrum' in list(data[group].keys()):
					enrg_spec[nn, :] = data[group]["EnergySpectrum"][:]
				nn += 1

	## Plot data
	fig  = plt.figure(figsize = (21, 9))
	ax1  = fig.add_subplot(131)
	im   = ax1.imshow(psi[0, :, :] / np.sqrt(np.mean(psi[0, :, :]**2)), norm=None, cmap="RdBu")
	cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad="2%")
	cb   = fig.colorbar(im, cax=cax1)
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax2  = fig.add_subplot(132)
	ax2.grid()
	ax2.plot(np.arange(1, int(Nx/3 + 1)), enrg_spec[0, 1:int(Nx/3 + 1)])
	ax3  = fig.add_subplot(133)
	ax3.plot(time[:0], tot_enrg[:0])
	ax3.set_xlim(time[0], time[-1])
	ax3.grid()
	for i in range(1, ndata):
		im.set_data(psi[i, :, :] / np.sqrt(np.mean(psi[i, :, :]**2)))
		ax2.plot(np.arange(1, int(Nx/3 + 1)), enrg_spec[i, 1:int(Nx/3 + 1)], 'b')
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.grid()
		ax3.plot(time[:i], tot_enrg[:i], 'b')
		ax3.set_xlim(time[0], time[-1])
		ax3.grid()

		plt.suptitle(r"Iter: {}".format(i))
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(1e-5)
		wait = input("Press " + tc.C + "ENTER" + tc.Rst + " to continue.")

	#######################
	##      CLEAN UP     ##
	#######################
	## Empty output directory
	run(["rm -r {}".format(output_dir + "*")], shell = True)
