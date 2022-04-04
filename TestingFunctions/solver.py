#!/usr/bin/env python3
import numpy as np
import pyfftw as fftw
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os
import sys
import getopt
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
### --------------------------- ###
### 	Function Definitions	###
### --------------------------- ###
def get_cmd_args(argv):

	"""
	Parses command line arguments
	"""

	## Create arguments class
	class cmd_args:

		"""
		Class for command line arguments
		"""

		def __init__(self, out_dir = "./Data/Test", Nx = 64, Ny = 64, nu = 1e-3, u0 = "HOPF_COLE", t0 = 0.0, dt = 1e-4, T = 1.0, print_steps = 10, visc = 1, pad = 3/2, tag = "HOPF-COLE-Test", mode = None):
			self.test_mode  = mode
			self.out_dir    = out_dir
			self.Nx         = Nx 
			self.Ny         = Ny
			self.nu         = nu
			self.u0         = u0 
			self.t0         = t0
			self.T          = T 
			self.pint_steps = print_steps
			self.visc       = visc
			self.pad        = pad
			self.tag        = tag
			self.dt 		= dt


	## Initialize class
	cargs = cmd_args()

	try:
		## Gather command line arguments
		opts, args = getopt.getopt(argv, "i:o::s:e:v:p:u:h:t:f:n:m:", ["mode=", "plot"])
	except:
		print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
		raise

	## Parse command line args
	for opt, arg in opts:
		if opt in ['-o']:
			## Read in output dir
			cargs.out_dir = str(arg)

		if opt in ['-s']:
			## Read in output dir
			cargs.t0 = float(arg)

		if opt in ['-e']:
			## Read in output dir
			cargs.T = float(arg)

		if opt in ['-n']:
			## Read in output dir
			cargs.Nx = int(arg)
			cargs.Ny = int(arg)

		if opt in ['-u']:
			## Read in output dir
			cargs.u0 = str(arg)

		if opt in ['-t']:
			## Read in output dir
			cargs.u0 = str(arg)

		if opt in ['-v']:
			## Read in output dir
			cargs.nu = float(arg)

		if opt in ['-p']:
			## Read in output dir
			cargs.print_steps = int(arg)

		if opt in ['-h']:
			## Read in output dir
			cargs.dt = float(arg)

		if opt in ['-m']:
			## Read in output dir
			cargs.visv = float(arg)

		if opt in ['--mode']:
			## Read in test_mode
			cargs.test_mode = str(arg)

	return cargs


def empty_real_array(shape, fft):

    if fft == "pyfftw":
        out = pyfftw.empty_aligned(shape, dtype='float64')
        out.flat[:] = 0.
        return out
    else:
        return np.zeros(shape, dtype='float64')


def empty_cmplx_array(shape, fft):

    if fft == "pyfftw":
        out = pyfftw.empty_aligned(shape, dtype='complex128')
        out.flat[:] = 0. + 0.*1.0j
        return out
    else:
        return np.zeros(shape, dtype='complex128')

def InitialConditions(Nx, Ny, Nyf, x, y, u0, mask):

	psi     = empty_real_array((Nx, Ny), "pyfft")
	psi_hat = empty_cmplx_array((Nx, Nyf), "pyfft")

	if u0 == "COLE_HOPF":

		for i in range(Nx):
			for j in range(Ny):
				## Get the solution to the heat equation
				theta = np.cos(x[i]) * np.sin(y[j]) * np.exp(-2.0 * NU * 0.0) + 2.0

				## Set the real velocity potential
				psi[i, j] = 2.0 * NU * np.log(theta)

		## Transform to Fourier space
		psi_hat[:, :] = psi_to_psih(psi) * filt_mask

	return psi_hat, psi

def NonlinearTerm(psi_hat, Nx, Ny, Nyf, kx, ky, mask):

	u_hat = empty_cmplx_array((Nx, Nyf), "pyfft")
	v_hat = empty_cmplx_array((Nx, Nyf), "pyfft")

	## Get grad psi in Fourier space
	u_hat[:, :] = 1j * kx[:, np.newaxis] * psi_hat[:, :]
	v_hat[:, :] = 1j * ky[:Nyf] * psi_hat[:, :]

	## Transform to Real Space
	u = uh_to_u(u_hat)
	v = vh_to_v(v_hat)

	## Multiply in Real Space
	u = u**2
	v = v**2

	## Transform Back
	u_hat = u_to_uh(u)
	v_hat = v_to_vh(v)

	## Compute the nonlinear term
	nonlin = 0.5 * (u_hat[:, :] + v_hat[:, :])

	return nonlin * mask


def TotalEnergy(psi_hat, k_sqr):

	enrg = 0.

	for i in range(psi_hat.shape[0]):
		for j in range(psi_hat.shape[1]):
			if j == 0 or j == psi_hat.shape[1] - 1:
				enrg += k_sqr[i, j] * np.absolute(psi_hat[i, j] * np.conjugate(psi_hat[i, j]))
			else:
				enrg += 2. * k_sqr[i, j] * np.absolute(psi_hat[i, j] * np.conjugate(psi_hat[i, j]))

	return enrg


def ExactSoln(x, y, t):

	exact_soln = empty_real_array((x.shape[0], y.shape[0]), "pyfft")

	for i in range(x.shape[0]):
		for j in range(y.shape[0]):

			theta = np.cos(x[i]) * np.sin(y[j]) * np.exp(-2.0 * NU * t) + 2.0

			exact_soln[i, j] = 2.0 * NU * np.log(theta)

	return exact_soln

### --------------------------- ###
### 			Main 			###
### --------------------------- ###
if __name__ == "__main__":

	### ---------------------------
	### System Parameters
	### ---------------------------
	cmdargs = get_cmd_args(sys.argv[1:])

	### ---------------------------
	### System Parameters
	### ---------------------------
	VISCOUS = 0
	TESTING = 1
	PADDED  = 0
	PAD = 3./2.


	## Space variables
	Nx    = cmdargs.Nx
	Ny    = cmdargs.Ny
	Nyf   = int(Ny / 2 + 1)
	Mx  = int(PAD * Nx)
	My  = int(PAD * Ny)
	Myf = int(PAD * Nyf)
	x, dx = np.linspace(0., 2. * np.pi, Nx, endpoint = False, retstep = True)
	y, dy = np.linspace(0., 2. * np.pi, Ny, endpoint = False, retstep = True)
	kx = np.append(np.arange(0, Nyf), np.linspace(-Ny//2 + 1, -1, Ny//2 - 1))
	ky = np.append(np.arange(0, Nyf), np.linspace(-Ny//2 + 1, -1, Ny//2 - 1))

	## File vars
	u0  = cmdargs.u0
	tag = cmdargs.tag


	## Pre-compute arrays
	k_sqr        = kx[:, np.newaxis]**2 + ky[:Nyf]**2    ## NOTE: This has y along the rows and x along the columns which is opposite to C code but values are exact same
	non_zer_indx = k_sqr != 0.0
	k_sqr_inv    = empty_cmplx_array((Nx, Nyf), "pyfft")
	k_sqr_inv[non_zer_indx] = 1. / k_sqr[non_zer_indx]


	## Dealias Mask
	if PADDED == 1:
	    # for easier slicing when padding
	    padder = np.ones(Mx, dtype = bool)
	    padder[int(Nx / 2):int(Nx * (PAD - 0.5)):] = False

	    filt_mask = empty_real_array((Nx, Nyf), "pyfft")
	    filt_mask.flat[:] = 1.
	else:
	    filt_mask = empty_real_array((Nx, Nyf), "pyfft")
	    Nk_23     = int(Nx / 3 + 1)
	    filt_indx = np.absolute(ky) < Nk_23 
	    filt_mask[filt_indx, :Nk_23] = 1.
	    padder = np.ones(Mx, dtype = bool)


	## Time variables
	t0 = cmdargs.t0
	t  = t0
	dt = cmdargs.dt
	T  = cmdargs.T
	print_iters = cmdargs.print_steps


	## Eqn parameters
	NU    = cmdargs.nu


	### ---------------------------
	### Allocate Memory
	### ---------------------------
	psi     = empty_real_array((Nx, Ny), "pyfft")
	u       = empty_real_array((Nx, Ny), "pyfft")
	v       = empty_real_array((Nx, Ny), "pyfft")
	v_hat   = empty_cmplx_array((Nx, Nyf), "pyfft")
	u_hat   = empty_cmplx_array((Nx, Nyf), "pyfft")
	psi_hat = empty_cmplx_array((Nx, Nyf), "pyfft")
	ww      = empty_real_array((Nx, Ny), "pyfft")
	wh      = empty_cmplx_array((Nx, Nyf), "pyfft")


	### ---------------------------
	### Set Up Transforms
	### ---------------------------
	psi_to_psih = fftw.FFTW(psi,  psi_hat, threads = 6, axes = (-2, -1))	
	psih_to_psi = fftw.FFTW(psi_hat,  psi, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
	u_to_uh     = fftw.FFTW(u,  u_hat, threads = 6, axes = (-2, -1))
	uh_to_u     = fftw.FFTW(u_hat,  u, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
	v_to_vh     = fftw.FFTW(v,  v_hat, threads = 6, axes = (-2, -1))
	vh_to_v 	= fftw.FFTW(v_hat,  v, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
	ifft2d  	= fftw.FFTW(wh,  ww, threads = 6, direction = 'FFTW_BACKWARD', axes = (-2, -1))
	
	### ---------------------------
	### Get Initial Conditions
	### ---------------------------
	psi_hat[:, :], psi[:, :] = InitialConditions(Nx, Ny, Nyf, x, y, u0, filt_mask)


	rhs = NonlinearTerm(psi_hat, Nx, Ny, Nyf, kx, ky, filt_mask)

	### ---------------------------
	### Write Initial Data to File
	### ---------------------------
	output_filename = cmdargs.out_dir + "/PyTestData_N[{},{}]_T[{}-{}]_NU[{:0.6f}]_u0[{}]_TAG[{}].h5".format(Nx, Ny, t0, T, NU, u0, tag)
	with h5py.File(output_filename, 'w') as out_file:
		print("Output file: " + tc.C + "{}".format(output_filename) + tc.Rst)
		## Create group for initial condition
		grp = out_file.create_group("/Iter_{:05}".format(0))
		## Write the initial data
		grp.create_dataset("psi", data = psi[:, :])
		grp.create_dataset("psi_hat", data = psi_hat[:, :])
		grp.create_dataset("RK1", data = rhs[:, :])
		## Write Space vars
		out_file.create_dataset("x", data = x[:])
		out_file.create_dataset("y", data = y[:])
		

	### ---------------------------
	### Write Initial Data to File
	### ---------------------------
	iters = 0
	while t <= T:

		## Stage 1
		RK1 = NonlinearTerm(psi_hat, Nx, Ny, Nyf, kx, ky, filt_mask)

		## Stage 2
		RK2 = NonlinearTerm(psi_hat + dt * 0.5 * RK1, Nx, Ny, Nyf, kx, ky, filt_mask)

		## Stage 3
		RK3 = NonlinearTerm(psi_hat + dt * 0.5 * RK2, Nx, Ny, Nyf, kx, ky, filt_mask)

		## Stage 4
		RK4 = NonlinearTerm(psi_hat + dt * RK3, Nx, Ny, Nyf, kx, ky, filt_mask)

		## Update Stage
		D = dt * NU * k_sqr
		psi_hat[:, :] = psi_hat[:, :] * ((2. - D) / (2. + D)) + (2. * dt / (2. + D)) * (1./6. * RK1 + 1./3. * RK2 + 1./3. * RK3 + 1./6. * RK4)

		t  += dt
		iters += 1

		if np.mod(iters, print_iters) == 0:
			## Transform to real space
			ph = psi_hat.copy()
			pw = ifft2d(ph)

			## Get error
			abs_err = np.absolute(pw - ExactSoln(x, y, t))
			# print("Iter: {} \t t: {:0.5f} \t dt: {:0.5f} \t KE: {:0.5f} \t ENS: {:0.5f}".format(iters, t, dt, TotalEnergy(psi_hat, k_sqr)))
			print("Iter: {} \t t: {:0.5f} \t dt: {:0.5f} \t KE: {:0.5f} \t L2: {:0.5g} \t Linf: {:0.5g}".format(iters, t, dt, TotalEnergy(psi_hat, k_sqr), np.linalg.norm(abs_err), np.linalg.norm(abs_err, ord = np.inf)))  ## L2_norm(abs_err, Nx, Ny), Linf_norm(abs_err)

			## Write to file
			with h5py.File(output_filename, 'a') as out_file:
				## Get group for current itreation
				grp = out_file.create_group("/Iter_{:05}".format(iters))
				## Write the initial data
				grp.create_dataset("psi", data = pw[:, :])
				grp.create_dataset("psi_hat", data = psi_hat[:, :])
				grp.create_dataset("RK1", data = RK1[:, :])    			
				grp.create_dataset("RK2", data = RK2[:, :])
				grp.create_dataset("RK3", data = RK3[:, :])
				grp.create_dataset("RK4", data = RK4[:, :])