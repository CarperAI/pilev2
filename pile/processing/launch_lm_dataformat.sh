#! /bin/bash

# convert group 2 subset to lm data format
# Set variables
DATA_PATH=/fsx/shared/pilev2_subset_25_group_2_1/tokenized
OUTPUT_PATH=/fsx/shared/pilev2/subset_group2_25_perc_lm_dataformat

# Create a list all of the directories in $DATA_PATH
datasets=$(ls $DATA_PATH)
mem=64GB
cpus=16
partition=cpu16
temp_sbatch=./temp_sbatch.slurm
# Loop through groups and submit slurm job for each dataset
for dataset in ${datasets[@]}; do   
    # list all files in $DATA_PATH/$dataset
    if [[ $dataset == fsx ]]; then
        # skip this dataset
        continue
    fi
    temp_dir=$DATA_PATH/$dataset
    basename_dataset=$(basename $dataset)
    basename_dataset=${basename_dataset%.*}
    echo $DATA_PATH/$dataset
    cat << HELP > $temp_sbatch
#!/bin/bash
# job name is the basename of the dataset and file
#SBATCH --job-name=$basename_dataset
#SBATCH --output=./logs/$basename_dataset.o
#SBATCH --error=./logs/$basename_dataset.e
#SBATCH --exclusive
#SBATCH --mem=$mem
#SBATCH --cpus-per-task=$cpus
#SBATCH --partition=$partition
#SBATCH --comment=carper
#SBATCH --export=ALL
# ===== END SLURM OPTIONS =====
source /fsx/home-erfan/miniconda3/bin/activate pilev2
cd /fsx/home-erfan/pilev2/pile/processing
python convert_to_lm_dataformat.py --data_dir $DATA_PATH/$dataset --output_dir $OUTPUT_PATH/$dataset
HELP
    sbatch $temp_sbatch
    rm $temp_sbatch
    # pause for 1 seconds
    sleep 1
done
