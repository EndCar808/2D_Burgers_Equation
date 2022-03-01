# Enda Carroll
# March 2022
# Makfile to compile programmes for Pseudopectral Solver for the 2D Burgers eqns

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
# Get the operating system:
# Need to strip leading and/or trailing whitespaces otherwise if statement wont work
OS = $(strip $(shell lsb_release -si))

# Set the operating system dependent compiler variables
ifeq ($(OS), Ubuntu)
	# CHIRP
	GCC := mpicc
	CCFLAG := -O3 -W -Wall -g
	LIBFLAG := -lfftw3_mpi -lfftw3 -lm -lhdf5 -lhdf5_hl #-fopenmp -lpthread -lfftw3_omp
	# GSLFLAG := -lgsl -lgslcblas
	# LAPACKFLAG := -llapacke -llapack -lblas -lgfortran
	INCDIR = -I/usr/include/hdf5/openmpi/  #-I/usr/include/gsl/
	LIBDIR = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi 
else ifeq ($(OS), RedHatEnterpriseServer)
	# SONIC
	GCC := mpicc
	CCFLAG := -O3 -W -Wall -g
	LIBFLAG := -lfftw3_mpi -lfftw3 -lm -lhdf5 -lhdf5_hl #-fopenmp -lpthread -lfftw3_omp
	INCDIR = -I/usr/include/hdf5/openmpi/ 
	LIBDIR = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi 
else ifeq ($(OS), CentOS)
	# KAY
	GCC := mpicc
	CCFLAG := -O3 -W -Wall -g
	LIBFLAG := -lfftw3_mpi -lfftw3 -lm -lhdf5 -lhdf5_hl #-fopenmp -lpthread -lfftw3_omp
	INCDIR = -I/ichec/packages/hdf5/gcc_mpi/1.12.0/include
	LIBDIR = -L/ichec/packages/hdf5/gcc_mpi/1.12.0/lib 
else
	$(error Unknown OS found, please check OS and add appropriate flags in Makefile)
endif

# ---------------------------------------------------------------------
#  Executable flags
# ---------------------------------------------------------------------
# Flags to be passed to the solver for normal runs
SOLVER_FLAGS := -D_VISCOUS -D__HYPER #-D__TRANSIENTS -D__ADAPTIVE_STEP -D__CFL_STEP -D_INVISCID

# Phase only flags to be passed to solver
PHASE_ONLY := -D_VISCOUS -D__HYPER -D__PHASE_ONLY -D__TRANSIENTS # -D_INVISCID

# Dataset Flags to be passed to the solver
DSET_FLAGS := -D__SYS_MEASURES #-D__ENST_FLUX -D__ENRG_FLUX -D__ENST_SPECT -D__ENRG_SPECT -D__ENST_FLUX_SPECT -D__ENRG_FLUX_SPECT -D__VORT_REAL -D__REALSPACE -D__MODES  
 
# Flags to be passed to the solver for testing
TEST_FLAGS := -D__INVISCID -D__TESTING -D__DEBUG #-D__ADAPTIVE_STEP -D__CFL_STEP -D__TRANSIENTS
# ---------------------------------------------------------------------
#  Directory creation
# ---------------------------------------------------------------------
OBJBIN := obj
OBJDIR := $(shell mkdir -p $(OBJBIN))

BIN    := bin
BINDIR := $(shell mkdir -p $(BIN))

OUT    := output
OUTDIR := $(shell mkdir -p $(OUT))

SRCDIR     := src
SRCFILE    := $(wildcard $(SRCDIR)/*.c)
SRCOBJ     := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%.o, $(SRCFILE))
SRCOBJ_PO  := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_phase_only.o, $(SRCFILE))
SRCOBJ_TST := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_test.o, $(SRCFILE))

DEPS = $(wildcard *.h) 

# ---------------------------------------------------------------------
#  Builds
# ---------------------------------------------------------------------
############
.PHONY: all
############


all: $(BIN)/solver $(BIN)/solver_phase_only $(BIN)/solver_test

####--------
#  Solver
####--------
solver: $(BIN)/solver 

$(BIN)/solver: $(SRCOBJ) $(DEPS)
	$(GCC) $(CCFLAG) $(SOLVER_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(SOLVER_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

####-----------
#  Phase Only 
####-----------
solver_phase_only: $(BIN)/solver_phase_only 

$(BIN)/solver_phase_only: $(SRCOBJ) $(DEPS)
	$(GCC) $(CCFLAG) $(PHASE_ONLY_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_phase_only.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(PHASE_ONLY_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

####--------
#  Testing
####--------
test: $(BIN)/solver_test

test_run: $(BIN)/solver_test
	mpirun -n 4 $(BIN)/solver_test -o "../Data/Test/" -n 128 -n 128 -s 0.0 -e 1.0 -h 0.0001 -i "TG_VORT" -v 1.0 -t "TG-TEST"

# Taylor Green IC
$(BIN)/solver_test: $(SRCOBJ_TG) $(DEPS)
	$(GCC) $(CCFLAG) $(TEST_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_test.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(TEST_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)
	
# ---------------------------------------------------------------------
#  Clean up
# ---------------------------------------------------------------------
clean:
	rm -rf $(OBJBIN)
	rm -rf $(BIN)
	rm -rf $(OUT)
# ---------------------------------------------------------------------
#  End of File
# ---------------------------------------------------------------------