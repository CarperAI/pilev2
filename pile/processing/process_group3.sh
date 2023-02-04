#! /bin/bash


# compression filtering for group 3
# Set variables
DATA_PATH=/fsx/shared/group1_splitted_after_dedup
OUTPUT_PATH=/fsx/shared/pilev2/group3_filtered

# Create a list all of the directories in $DATA_PATH
datasets=$(ls $DATA_PATH)
mem=256GB
cpus=64
partition=cpu64
temp_sbatch=./temp_sbatch.slurm
# Loop through groups and submit slurm job for each dataset
for dataset in ${datasets[@]}; do   
    # list all files in $DATA_PATH/$dataset
    temp_dir=$DATA_PATH/$dataset
    basename_dataset=$(basename $dataset)
    basename_dataset=${basename_dataset%.*}
    echo $DATA_PATH/$dataset/$file
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
python compression_filtering.py --data_dir $DATA_PATH/$dataset --output_dir $OUTPUT_PATH/$basename_dataset --do_shard

HELP
    sbatch $temp_sbatch
    rm $temp_sbatch
    # pause for 10 seconds
    sleep 2
done
