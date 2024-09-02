#!/bin/bash

# User info
#SBATCH --mail-user=simone.poncioni@unibe.ch
#SBATCH --mail-type=begin,end,fail

# Job name
#SBATCH --job-name="hfe_pipeline"

# Runtime and memory
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=100G
#SBATCH --array=1-5%5

# Workdir
#SBATCH --chdir=/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/HFE
#SBATCH --output=out/hfe_%A_%a.out
#SBATCH --error=out/hfe_%A_%a.err

##############################################################################################################
### Load modules
HPC_WORKSPACE=hpc_abaqus module load Workspace

unset SLURM_GTIDS

### greyscale_filenames.txt contains lines with 1 greyscale_filename per line.
greyscale_filenames=/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/HFE/00_ORIGAIM/filenames.txt

###    Line <i> contains greyscale_filename for run <i>
# Get greyscale_filename                                                                                                                              
greyscale_filename=$(cat $greyscale_filenames | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')


### Zero pad the task ID to match the numbering of the input files
n=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

# Run command
srun apptainer exec /storage/workspaces/artorg_msb/hpc_abaqus/poncioni/apptainer/hfe_development_ifort.sif \
/bin/bash -c "source /opt/intel/oneapi/setvars.sh && source /opt/miniconda/etc/profile.d/conda.sh && conda activate hfe-essentials && python 02_CODE/src/pipeline_runner.py simulations.grayscale_filenames=$greyscale_filename"

