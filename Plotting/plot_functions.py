#!/usr/bin/env python    

## Author: Enda Carroll
## Date: Sept 2021
## Info: Plotting functions for solver data


#######################
##  Library Imports  ##
#######################
import numpy as np
import sys
import os
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = 'Computer Modern Roman'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from subprocess import Popen, PIPE
from itertools import zip_longest
from numba import njit

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



#######################################
##       VIDEO MAKING FUNCTIONS      ##
#######################################
def run_plotting_snaps_parallel(args_list, num_procs = 20):

    """
    Runs plotting command functions in parallel.

    args_list : list
                - List of commands that will be used to run in parallel
    num_procs : int
                - The number of process to use in parallel
    """

    ## Define the number of processes to use
    proc_lim = num_procs

    ## Get the group of arguments to run
    group_args = args_list * proc_lim

    ## Loop of grouped iterable
    for procs in zip_longest(*groups_args): 
        pipes     = []
        processes = []
        for p in filter(None, procs):
            recv, send = mprocs.Pipe(False)
            processes.append(p)
            pipes.append(recv)
            p.start()

        for process in processes:
            process.join()



def make_video(fileanme, snapname, fps = 30):

    """
    Runs the ffmpeg command to make videos from snaps

    filename : string
                - the filename path of the video
    snapname : string
                - The name of the snapshot files to use 
    fps      : int
                - The number of frames per second to use in making the video
    """

    ## Contruct input file names and path
    snaps     = snapname    
    videoname = fileanme

    ## Contruct the ffmpeg command
    cmd = "ffmpeg -y -r {} -f image2 -s 1920x1080 -i {} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(fps, snaps, videoname)

    ## Run command
    process = Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)
    [runCodeOutput, runCodeErr] = process.communicate()
    print(runCodeOutput)
    print(runCodeErr)
    process.wait()

    ## Prin summary of timmings to screen
    print("\n" + tc.Y + "Finished making video..." + tc.Rst)
    print("Video Location: " + tc.C + videoName + tc.Rst + "\n")


#######################################
##       SUMMARY SNAP FUNCTIONS      ##
#######################################
def plot_summary_snaps(psi, enrg_spec):