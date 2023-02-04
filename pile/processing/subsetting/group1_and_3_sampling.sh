#! /bin/bash

# Set variables
GROUPED_DUPS=./groups_1_and_3.json
OUTPUT_PATH=/fsx/shared/pilev2_subset_25_group1_and_group3

# Read groups.json
group1=$(jq -r '.group_1[]' $GROUPED_DUPS)
group3=$(jq -r '.group_3[]' $GROUPED_DUPS)


# Concatenate group1 and group3
all_groups="$group3" # 

# Loop through groups and submit slurm job for each dataset
for dataset in $all_groups; do

    basename_dataset=$(basename $dataset)

    # if $dataset starts with PileV2 or contains C4 then append non_local_dedup to $DATA_PATH
    if [[ $dataset == *PubMed* ]]; then
        mem=256GB
        cpus=64
        partition=cpu64
    elif  [[ $dataset == *S2ORC* ]]; then
        mem=512GB
        cpus=128
        partition=cpu128

    elif  [[ $dataset == *the_stack_filtered_ext* ]]; then
        mem=512GB
        cpus=64
        partition=cpu128
        basename_dataset=the_stack_filtered_ext_$basename_dataset
    else
        mem=128GB
        cpus=32
        partition=cpu32
    fi
    temp_sbatch=./temp_sbatch.slurm
    cat << HELP > $temp_sbatch
#!/bin/bash
#SBATCH --job-name=$basename_dataset
#SBATCH --output=$OUTPUT_PATH/sampling_logs/$basename_dataset.o
#SBATCH --error=$OUTPUT_PATH/sampling_logs/$basename_dataset.e
#SBATCH --exclusive
#SBATCH --mem=$mem
#SBATCH --cpus-per-task=$cpus
#SBATCH --partition=$partition
#SBATCH --comment=carper
#SBATCH --export=ALL
# ===== END SLURM OPTIONS =====
source /fsx/home-erfan/miniconda3/bin/activate pilev2
cd /fsx/home-erfan/pilev2/pile/processing/subsetting
python subset_pilev2_and_tokenize.py --input_dir $dataset --output_dir $OUTPUT_PATH/$basename_dataset --num_workers "$(($cpus-3))"

HELP
    sbatch $temp_sbatch
    rm $temp_sbatch
done


