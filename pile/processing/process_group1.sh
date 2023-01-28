#! /bin/bash

# Set variables
DATA_PATH=/fsx/shared/group1_splitted_after_dedup
OUTPUT_PATH=/fsx/shared/pilev2/group1_filtered

# Create a list all of the directories in $DATA_PATH
datasets=$(ls $DATA_PATH)

mem=32GB
cpus=12
partition=cpu64
temp_sbatch=./temp_sbatch.slurm
# Loop through groups and submit slurm job for each dataset
for dataset in $datasets; do   
    # list all files in $DATA_PATH/$dataset
    files=$(ls $DATA_PATH/$dataset)
    # Loop through files in $DATA_PATH/$dataset
    for file in $files; do
        cat << HELP > $temp_sbatch
        #!/bin/bash
        # job name is the basename of the dataset and file
        #SBATCH --job-name=$basename_dataset-$file
        #SBATCH --output=./logs/$basename_dataset-$file.o
        #SBATCH --error=./logs/$basename_dataset-$file.e
        #SBATCH --exclusive
        #SBATCH --mem=$mem
        #SBATCH --cpus-per-task=$cpus
        #SBATCH --partition=$partition
        #SBATCH --comment=carper
        #SBATCH --export=ALL
        # ===== END SLURM OPTIONS =====
        source /fsx/home-erfan/miniconda3/bin/activate pilev2
        cd /fsx/home-erfan/pilev2/pile/processing
        python group_1.py --data_dir $DATA_PATH/$dataset/$file --output_dir $OUTPUT_PATH/$basename_dataset/$file

        HELP
            sbatch $temp_sbatch
            rm $temp_sbatch

done
done