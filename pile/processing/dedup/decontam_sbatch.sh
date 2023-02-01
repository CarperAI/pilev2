#! /bin/bash

# Set variables
DATA_PATH=/fsx/shared/pilev2/group1_filtered
OUTPUT_PATH=/fsx/shared/pilev2/decontam/other_benchmarks

# Create a list all of the directories in $DATA_PATH
datasets=$(ls $DATA_PATH)
mem=32GB
cpus=16
partition=cpu16
temp_sbatch=./temp_sbatch.slurm
mkdir ./logs
export HF_ACCESS_TOKEN=hf_paJlpPLHECIqGJuDEMOwxnsmsbjhYRTqJH
# Loop through groups and submit slurm job for each dataset
for dataset in ${datasets[@]}; do
    # skip if the dataset is named "arXiv" and "PubMed" and "USPTO"
    if [ $dataset == "arXiv" ] || [ $dataset == "PubMed" ] || [ $dataset == "USPTO" ]; then
        continue
    fi
    if [ $dataset != "UbuntuIRC" ]; then
        continue
    fi
    
    echo $DATA_PATH/$dataset
    basename_dataset=$(basename $dataset)
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
source /fsx/nathan/miniconda3/bin/activate pilev2
cd /fsx/home-nathan/work/pilev2/pile/processing/dedup
python decontainmentation.py \
    --data_dir $DATA_PATH/$dataset \
    --output_dir $OUTPUT_PATH

HELP
    sbatch $temp_sbatch
    rm $temp_sbatch
    # pause for 10 seconds
    sleep 10
done