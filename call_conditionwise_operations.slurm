#!/bin/bash
## call_filewise_operations.slurm
#SBATCH --job-name=process_file
#SBATCH --output=logs/CONDITIONWISE_output_%A_%a.log
##SBATCH --error=logs/error_%A_%a.log
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4             # number of cores/threads requested.
#SBATCH --partition=20                 # partition (queue) to use
##SBATCH --mail-type=ALL               # send email on job start/finish.
##SBATCH --mail-user=gschliss@wi.mit.edu

# Access environment variables exported from Python

echo "Settings file: $PKL_PATH"
echo "Condition:     $CONDITION"
echo "Code from:     $CODE_DIRECTORY"

# Example: pass to your processing script
/lab/tambora_li/Gavin/conda_env_dir/saspt_env/bin/python3 $CODE_DIRECTORY/run_conditionwise.py --condition "$CONDITION" --settingsPKL "$PKL_PATH"
