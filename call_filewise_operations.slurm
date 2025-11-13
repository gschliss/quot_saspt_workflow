#!/bin/bash
## call_filewise_operations.slurm
#SBATCH --job-name=process_file
#SBATCH --output=logs/FILEWISE_output_%A_%a.log
##SBATCH --error=logs/error_%A_%a.log
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1             # number of cores/threads requested.
#SBATCH --partition=20                 # partition (queue) to use
##SBATCH --mail-type=ALL               # send email on job start/finish.
##SBATCH --mail-user=gschliss@wi.mit.edu

# Access environment variables exported from Python

IFS=":" read -r -a ITEMS_ARR <<< "$ITEM_STRING"
item=${ITEMS_ARR[$SLURM_ARRAY_TASK_ID]}

echo "ND2 file:      $item"
echo "Settings file: $PKL_PATH"
echo "Condition:     $CONDITION"
echo "Code from:     $CODE_DIRECTORY"

# Example: pass to your processing script
/lab/tambora_li/Gavin/conda_env_dir/saspt_env/bin/python3 $CODE_DIRECTORY/run_filewise.py --condition "$CONDITION" --inputfile "$item" --settingsPKL "$PKL_PATH"
