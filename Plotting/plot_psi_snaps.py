#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros
######################
##  Library Imports ##  
######################
import matplotlib as mpl
# mpl.use('TkAgg') # Use this backend for displaying plots in window
mpl.use('Agg') # Use this backend for writing plots to file
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import os
import getopt
from itertools import zip_longest
import multiprocessing as mprocs
import time as TIME
from subprocess import Popen, PIPE
from numba import njit
import pyfftw as fftw

from functions import tc, sim_data, import_data, import_spectra_data
from plot_functions import run_plotting_snaps_parallel, make_video, plot_summary_snaps


###############################
##       FUNCTION DEFS       ##
###############################
def parse_cml(argv):

    """
    Parses command line arguments
    """

    ## Create arguments class
    class cmd_args:

        """
        Class for command line arguments
        """
        
        def __init__(self, in_dir = None, out_dir = None, phase_dir = None, main_file = None, spec_file = None, summ_snap = False, phase_snap = False, parallel = False, plotting = False, video = False, use_post = False, post_file = None):
            self.spec_file  = spec_file
            self.main_file  = main_file
            self.post_file  = post_file
            self.in_dir     = in_dir
            self.out_dir    = out_dir
            self.phase_dir  = phase_dir
            self.summ_snap  = summ_snap
            self.phase_snap = phase_snap
            self.parallel   = parallel
            self.plotting   = plotting
            self.video      = video 
            self.use_post   = use_post

    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:m:s:f:", ["s_snap", "p_snap", "par", "plot", "vid", "use_post"])
    except:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        sys.exit()

    ## Parse command line args
    for opt, arg in opts:
        
        if opt in ['-i']:
            ## Read input directory
            cargs.in_dir = str(arg)
            print("Input Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

        elif opt in ['-o']:
            ## Read in output directory
            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)


        elif opt in ['-m']:
            ## Read in main file
            cargs.main_file = str(arg)
            print("Main File: " + tc.C + "{}".format(cargs.main_file) + tc.Rst)

        elif opt in ['-s']:
            ## Read in spectra file
            cargs.spec_file = str(arg)
            print("Spectra File: " + tc.C + "{}".format(cargs.spec_file) + tc.Rst)

        elif opt in ['--s_snap']:
            ## Read in summary snaps indicator
            cargs.summ_snap = True

            ## Make summary snaps output directory 
            cargs.out_dir = cargs.in_dir + "SNAPS/"
            if os.path.isdir(cargs.out_dir) != True:
                print("Making folder:" + tc.C + " SNAPS/" + tc.Rst)
                os.mkdir(cargs.out_dir)
            print("Output Folder: "+ tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        elif opt in ['--p_snap']:
            ## Read in phase_snaps indicator
            cargs.phase_snap = True

            ## Make phase snaps output directory
            cargs.phase_dir = cargs.in_dir + "PHASE_SNAPS/"
            if os.path.isdir(cargs.phase_dir) != True:
                print("Making folder:" + tc.C + " PHASE_SNAPS/" + tc.Rst)
                os.mkdir(cargs.phase_dir)
            print("Phases Output Folder: "+ tc.C + "{}".format(cargs.phase_dir) + tc.Rst)

        elif opt in ['--use_post']:
            ## Read in use of post processing indicator
            cargs.use_post = True

        elif opt in ['-f']:
            ## Read post processing file
            cargs.post_file = str(arg)

            print("Post Processing File: "+ tc.C + "{}".format(cargs.post_file) + tc.Rst)

        elif opt in ['--par']:
            ## Read in parallel indicator
            cargs.parallel = True

        elif opt in ['--plot']:
            ## Read in plotting indicator
            cargs.plotting = True

        elif opt in ['--vid']:
            ## Read in spectra file
            cargs.video = True

    return cargs


######################
##       MAIN       ##
######################
if __name__ == '__main__':
  
    # -------------------------------------
    ## --------- Parse Commnad Line
    # -------------------------------------
    cmdargs = parse_cml(sys.argv[1:]) 
    method  = "default"
    

    # -----------------------------------------
    ## --------  Read In data
    # -----------------------------------------
    ## Read in simulation parameters
    sys_vars = sim_data(cmdargs.in_dir, method)

    ## Read in solver data
    run_data = import_data(cmdargs.in_dir, sys_vars, method)


    ## Read in spectra data
    if cmdargs.spec_file == None and os.path.isfile(cmdargs.in_dir + "Spectra_HDF_Data.h5") == True:
        ## Read spectra file in normal mode
        spec_data = import_spectra_data(cmdargs.in_dir, sys_vars, method)
    elif cmdargs.spec_file == None and os.path.isfile(cmdargs.in_dir + "Spectra_HDF_Data.h5") != True:
        pass
    else:
        ## Read in spectra file in file only mode
        spec_data = import_spectra_data(cmdargs.spec_file, sys_vars, method)


    # -----------------------------------------
    ## ------ Plot Snaps
    # -----------------------------------------
    if cmdargs.plotting:
        
        ## Start timer
        start = TIME.perf_counter()
        print("\n" + tc.Y + "Printing Snaps..." + tc.Rst + "Total Snaps to Print: [" + tc.C + "{}".format(sys_vars.ndata) + tc.Rst + "]")
        
        ## Print main summary snaps
        if cmdargs.summ_snap:
            print("\n" + tc.Y + "Printing Summary Snaps..." + tc.Rst)
            if cmdargs.parallel:

                ## Generate list of commands to run
                arg_list = [(mprocs.Process(
                    target = plot_summary_snaps, 
                    args = (cmdargs.out_dir, i, 
                            run_data.psi[i, :, :], 
                            run_data.time, 
                            run_data.x, 
                            run_data.y, 
                            spec_data.enrg_spec[i, :], 
                            np.absolute(run_data.psi[i, :, :] - run_data.exact_soln[i, :, :]), 
                            run_data.tot_enrg[:i], 
                            run_data.tot_div_sqr[:i], 
                            run_data.tot_uv[:i], 
                            run_data.tot_usqr_vsqr[:i], 
                            sys_vars.Nx)) for i in range(run_data.psi.shape[0]))]

                ## Run commmands in parallel
                run_plotting_snaps_parallel(arg_list, num_procs = 20)

            else:
                ## Loop over snapshots
                for i in range(sys_vars.ndata):
                    plot_summary_snaps(cmdargs.out_dir, i, run_data.psi[i, :, :], run_data.time, run_data.x, run_data.y, spec_data.enrg_spec[i, :], np.absolute(run_data.psi[i, :, :] - run_data.exact_soln[i, :, :]), run_data.tot_enrg[:i], run_data.tot_div_sqr[:i], run_data.tot_uv[:i], run_data.tot_usqr_vsqr[:i], sys_vars.Nx)


        ## End timer
        end = TIME.perf_counter()
        plot_time = end - start
        print("\n" + tc.Y + "Finished Plotting..." + tc.Rst)
        print("\n\nPlotting Time: " + tc.C + "{:5.8f}s\n\n".format(plot_time) + tc.Rst)


    #------------------------------------
    # ----- Make Video
    #------------------------------------
    if cmdargs.video:

        ## Start timer
        start = TIME.perf_counter()

        if cmdargs.summ_snap:

            ## Make the video from the summary snaps
            filename = cmdargs.out_dir + "2D_Burgers_N[{},{}]_u0[{}].mp4".format(sys_vars.Nx, sys_vars.Ny, sys_vars.u0)
            snapname = cmdargs.out_dir + "SNAP_%05d.png"
            make_video(filename, snapname, fps = 30)

        ## Start timer
        end = TIME.perf_counter()

        ## Print summary of timmings to screen
        if cmdargs.plotting:
            print("\n\nPlotting Time:" + tc.C + " {:5.8f}s\n\n".format(plot_time) + tc.Rst)
        if cmdargs.video:
            print("Movie Time:" + tc.C + " {:5.8f}s\n\n".format(end - start) + tc.Rst)