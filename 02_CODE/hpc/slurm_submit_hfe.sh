#!/bin/bash

# User info
#SBATCH --mail-user=simone.poncioni@unibe.ch
#SBATCH --mail-type=begin,end,fail

# Job name
#SBATCH --job-name="hfe_pipeline_test"

# Runtime and memory
#SBATCH --partition=epyc2,bdw
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=64G

#SBATCH --output=%x.out 
#SBATCH --error=%x.err
#SBATCH --chdir=/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/HFE

# Load modules
HPC_WORKSPACE=hpc_abaqus module load Workspace
module load GCCcore/12.3.0
unset SLURM_GTIDS

# Run command
srun apptainer exec \
/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/apptainer/hfe_development.sif \
/bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && \
conda activate hfe-essentials && \
python 02_CODE/src/pipeline_runner.py"