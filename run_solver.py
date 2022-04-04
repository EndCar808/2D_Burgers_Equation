#!/usr/bin/env python    
#######################
##  LIBRARY IMPORTS  ##
#######################
from configparser import ConfigParser
import numpy as np
import os
import sys
import distutils.util as utils
from datetime import datetime
from collections.abc import Iterable
from itertools import zip_longest
from subprocess import Popen, PIPE
from Plotting.functions import tc
############################
##  FUNCTION DEFINITIONS  ##
############################
def run_command(cmdList, num_procs, solver_output, solver_error, collect_data):

    '''
    Runs the provided list of commands in the terminal in parallel
    '''

    ## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
    groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmdList)] * num_procs 

    ## Loop through grouped iterable
    for processes in zip_longest(*groups): 
        for proc in filter(None, processes): # filters out 'None' fill values if num_procs does not divide evenly into cmdList
            ## Print command to screen
            print("Executing the following command:\n\t" + tc.C + "{}".format(proc.args[0]) + tc.Rst)
            
            # Communicate with process to retrive output and error
            [run_CodeOutput, run_CodeErr] = proc.communicate()

            # Append to output and error objects
            if collect_data:
                solver_output.append(run_CodeOutput)
                solver_error.append(run_CodeErr)
            
            ## Print both to screen
            print(run_CodeOutput)
            print(run_CodeErr)

            ## Wait until all finished
            proc.wait()

def write_command_output(config_file, cmdList, solver_output, solver_error):

    '''
    Writes command output data to file
    '''

    # Get data and time
    now = datetime.now()
    d_t = now.strftime("%d%b%Y_%H:%M:%S")

    # Write output to file
    with open("Data/ParallelRunsDump/par_run_solver_output_{}_{}.txt".format(config_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
        for item in solver_output:
            file.write("%s\n" % item)

    # Write error to file
    with open("Data/ParallelRunsDump/par_run_solver_error_{}_{}.txt".format(config_file.lstrip('InitFiles/').rstrip(".ini"), d_t), "w") as file:
        for i, item in enumerate(solver_error):
            file.write("%s\n" % cmdList[i])
            file.write("%s\n" % item)

######################
##       MAIN       ##
######################
if __name__ == '__main__':
    
    #########################
    ##  READ COMMAND LINE  ##
    #########################
    ## Read in .ini file
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        print("Input configuration file: " + tc.C + config_file + tc.Rst)
    else:
        print("[" + tc.R + "ERROR" + tc.Rst + "] --- No config file provided...Exiting!")
        sys.exit()

    ##########################
    ##  DEFAULT PARAMETERS  ##
    ##########################
    ## Space variables
    Nx = 128
    Ny = 128
    Nk = int(Ny / 2 + 1)
    ## System parameters
    nu             = 0.001
    hypervisc      = int(1) 
    ekmn_alpha     = 0.0
    ekmn_hypo_diff = -int(2)
    ## Time parameters
    t0        = 0.0
    T         = 1.0
    dt        = 1e-3
    step_type = True
    cfl       = np.sqrt(3)
    ## Solver parameters
    ic         = "DECAY"
    forcing    = "NONE"
    force_k    = 0
    save_every = 100
    ## Directory/File parameters
    input_dir       = "NONE"
    output_dir      = "./Data/Tmp/"
    file_only_mode  = False
    solver_tag      = "Decay-Test"
    post_input_dir  = output_dir
    post_output_dir = output_dir
    ## Job parameters
    executable                  = "Solver/bin/main"
    plot_options                = "--full_snap --base_snap --plot --vid"
    plotting                    = True
    solver                      = True
    postprocessing              = True
    solver_procs                = 4
    num_solver_job_threads      = 1
    num_postprocess_job_threads = 1


    #########################
    ##  PARSE CONFIG FILE  ##
    #########################
    ## Create parser instance
    parser = ConfigParser()

    ## Read in config file
    parser.read(config_file)

    ## Create list objects
    Nx         = []
    Ny         = []
    Nk         = []
    nu         = []
    ic         = []
    T          = []
    dt         = []
    cfl        = []
    solver_tag = []

    ## Parse input parameters
    for section in parser.sections():
        if section in ['SYSTEM']:
            if 'nx' in parser[section]:
                for n in parser[section]['nx'].lstrip('[').rstrip(']').split(', '):
                    Nx.append(int(n))
            if 'ny' in parser[section]:    
                for n in parser[section]['ny'].lstrip('[').rstrip(']').split(', '):
                    Ny.append(int(n))
            if 'nk' in parser[section]:
                for n in parser[section]['nk'].lstrip('[').rstrip(']').split(', '):
                    Nk.append(int(n))
            if 'viscosity' in parser[section]:
                for n in parser[section]['viscosity'].lstrip('[').rstrip(']').split(', '):
                    nu.append(float(n))
            if 'drag_coefficient' in parser[section]:
                ekmn_alpha = float(parser[section]['drag_coefficient'])
        if section in ['SOLVER']:
            if 'initial_condition' in parser[section]:
                for n in parser[section]['initial_condition'].lstrip('[').rstrip(']').split(', '):
                    ic.append(str(parser[section]['initial_condition']))
            if 'forcing' in parser[section]:
                forcing = str(parser[section]['forcing'])
            if 'forcing_wavenumber' in parser[section]:
                force_k = int(parser[section]['forcing_wavenumber'])
            if 'save_data_every' in parser[section]:
                save_every = int(parser[section]['save_data_every'])
        if section in ['TIME']:
            if 'end_time' in parser[section]:
                for n in parser[section]['end_time'].lstrip('[').rstrip(']').split(', '):
                    T.append(float(parser[section]['end_time']))
            if 'timestep' in parser[section]:
                for n in parser[section]['timestep'].lstrip('[').rstrip(']').split(', '):
                    dt.append(float(parser[section]['timestep']))
            if 'cfl' in parser[section]:
                for n in parser[section]['cfl'].lstrip('[').rstrip(']').split(', '):
                    cfl.append(float(parser[section]['cfl']))
            if 'start_time' in parser[section]:
                t0 = float(parser[section]['start_time'])
            step_type = bool(utils.strtobool(parser[section]['adaptive_step_type']))
        if section in ['DIRECTORIES']:
            if 'solver_input_dir' in parser[section]:
                input_dir = str(parser[section]['solver_input_dir'])
            if 'solver_output_dir' in parser[section]:
                output_dir = str(parser[section]['solver_output_dir'])
            if 'solver_tag' in parser[section]:
                for n in parser[section]['solver_tag'].lstrip('[').rstrip(']').split(', '):
                    solver_tag.append(str(parser[section]['solver_tag']))
            if 'post_input_dir' in parser[section]:
                post_input_dir = str(parser[section]['post_input_dir'])
            if 'post_output_dir' in parser[section]:
                post_output_dir = str(parser[section]['post_output_dir'])
            if 'solver_file_only_mode' in parser[section]:
                file_only_mode = bool(utils.strtobool(parser[section]['solver_file_only_mode']))
            if 'system_tag' in parser[section]:
                system_tag = str(parser[section]['system_tag'])
        if section in ['JOB']:
            if 'executable' in parser[section]:
                executable = str(parser[section]['executable'])
            if 'plotting' in parser[section]:
                plotting = str(parser[section]['plotting'])
            if 'plot_script' in parser[section]:
                plot_script = str(parser[section]['plot_script'])
            if 'plot_options' in parser[section]:
                plot_options = str(parser[section]['plot_options'])
            if 'call_solver' in parser[section]:
                solver = bool(utils.strtobool(parser[section]['call_solver']))
            if 'call_postprocessing' in parser[section]:
                postprocessing = bool(utils.strtobool(parser[section]['call_postprocessing']))
            if 'solver_procs' in parser[section]:
                solver_procs = int(parser[section]['solver_procs'])
            if 'collect_data' in parser[section]:
                collect_data = bool(utils.strtobool(parser[section]['collect_data']))
            if 'num_solver_job_threads' in parser[section]:
                num_solver_job_threads = int(parser[section]['num_solver_job_threads'])
            if 'num_postprocess_job_threads' in parser[section]:
                num_postprocess_job_threads = int(parser[section]['num_postprocess_job_threads'])
            if 'num_plotting_job_threads' in parser[section]:
                num_plotting_job_threads = int(parser[section]['num_plotting_job_threads'])
            

    #########################
    ##      RUN SOLVER     ##
    #########################
    if solver:

        ## Get the number of processes to launch
        proc_limit = num_solver_job_threads
        print("Number of Solver Processes Created = [" + tc.C + "{}".format(proc_limit) + tc.Rst + "]")

        # Create output objects to store process error and output
        if collect_data:
            solver_output = []
            solver_error  = []

        ## Generate command list 
        cmd_list = [["mpirun -n {} {} -o {} -n {} -n {} -s {:3.1f} -e {:3.1f} -c {:1.6f} -h {:1.6f} -v {:1.10f} -v {} -a {:1.6f} -a {} -i {} -t {} -f {} -f {} -p {}".format(
                        solver_procs, 
                        executable, 
                        output_dir, 
                        nx, ny, 
                        t0, t, 
                        c, h, 
                        v, hypervisc,
                        ekmn_alpha, int(ekmn_hypo_diff), 
                        u0, 
                        s_tag, 
                        forcing, force_k, 
                        save_every)] for nx, ny in zip(Nx, Ny) for t in T for h in dt for u0 in ic for v in nu for c in cfl for s_tag in solver_tag]

        ## Run command
        if collect_data:
            run_command(cmd_list, proc_limit, solver_output, solver_error, collect_data)
        else:
            run_command(cmd_list, proc_limit, [], [], collect_data)

        ## Write command output
        if collect_data:
            write_command_output(config_file, cmd_list, solver_output, solver_error)

##################################
##      RUN POST PROCESSING     ##
##################################
if postprocessing:
    
    ## Get the number of processes to launch
    proc_limit = num_postprocess_job_threads
    print("Number of Post Processing Processes Created = [" + tc.C + "{}".format(proc_limit) + tc.Rst + "]")

    # Create output objects to store process error and output
    if collect_data:
        post_output = []
        post_error  = []

    ## Generate command list 
    cmd_list = [["PostProcessing/bin/main -i {} -o {}".format(
                    post_input_dir + "N[{},{}]_T[{}-{}]_NU[{:1.6f}]_CFL[{:1.2f}]_u0[{}]_TAG[{}]/".format(nx, ny, int(t0), int(t), v, c, u0, s_tag), 
                    post_output_dir + "N[{},{}]_T[{}-{}]_NU[{:1.6f}]_CFL[{:1.2f}]_u0[{}]_TAG[{}]/".format(nx, ny, int(t0), int(t), v, c, u0, s_tag))] 
                    for nx, ny in zip(Nx, Ny) for t in T for v in nu for c in cfl for u0 in ic for s_tag in solver_tag]
    
    ## Run command
    if collect_data:
        run_command(cmd_list, proc_limit, solver_output, solver_error, collect_data)
    else:
        run_command(cmd_list, proc_limit, [], [], collect_data)

    ## Write command output
    if collect_data:
        write_command_output(config_file, cmd_list, solver_output, solver_error)


###########################
##      RUN PLOTTING     ##
###########################
if plotting:
    
    ## Get the number of processes to launch
    proc_limit = num_plotting_job_threads
    print("Number of Post Processing Processes Created = [" + tc.C + "{}".format(proc_limit) + tc.Rst + "]")

    # Create output objects to store process error and output
    if collect_data:
        plot_output = []
        plot_error  = []

    ## Generate command list 
    cmd_list = [["python3 {} -i {} {}".format(
                    plot_script, 
                    post_input_dir + "N[{},{}]_T[{}-{}]_NU[{:1.6f}]_CFL[{:1.2f}]_u0[{}]_TAG[{}]/".format(nx, ny, int(t0), int(t), v, c, u0, s_tag), 
                    plot_options)] 
                    for nx, ny in zip(Nx, Ny) for t in T for v in nu for c in cfl for u0 in ic for s_tag in solver_tag]

    ## Run command
    if collect_data:
        run_command(cmd_list, proc_limit, solver_output, solver_error, collect_data)
    else:
        run_command(cmd_list, proc_limit, [], [], collect_data)

    ## Write command output
    if collect_data:
        write_command_output(config_file, cmd_list, solver_output, solver_error)