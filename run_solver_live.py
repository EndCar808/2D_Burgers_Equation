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



######################
##       MAIN       ##
######################
if __name__ == '__main__':

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
	T              = 1.0
	c              = 0.7
	h              = 1e-3
	v              = 1e2
	hypervisc      = int(1)
	ekmn_alpha     = 0.0
	ekmn_hypo_diff = -int(2)
	u0             = "TG_VEL_POT"
	s_tag          = "TG-Live"
	forcing 	   = 0
	force_k        = 0
	ndata          = 50
	save_every     = int((T - t0) / (h * ndata))

	## Clean output directory
	run(["rm -r {}".format(output_dir + "*")], shell = True)

	#########################
	##      RUN SOLVER     ##
	#########################
	## Generate command list 
	cmd_list = ["mpirun -n {} {} -o {} -n {} -n {} -s {:3.1f} -e {:3.1f} -c {:1.6f} -h {:1.6f} -v {:1.10f} -v {} -a {:1.6f} -a {} -i {} -t {} -f {} -f {} -p {}".format(
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
	data_file = output_dir + os.listdir(output_dir)[0] + "/Main_HDF_Data.h5"
	print("\nData File:" + tc.C + "{}".format(data_file) + tc.Rst)
	with h5py.File(data_file, 'r') as data:
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

	## Plot data
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	im  = ax.imshow(psi[0, :, :], norm=None, cmap="RdBu")
	cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")
	cb  = fig.colorbar(im, cax=cax)
	ax.set_xticks([])
	ax.set_yticks([])
	for i in range(1, ndata):
		im.set_data(psi, norm=None, cmap="RdBu")
		plt.title(r"Iter: {}".format(i))
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(1e-9)

	#######################
	##      CLEAN UP     ##
	#######################
	## Empty output directory
	run(["rm -r {}".format(output_dir + "*")], shell = True)
