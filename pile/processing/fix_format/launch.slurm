#!/bin/bash
#SBATCH --job-name=formatter
#SBATCH --output=./logs/PubMed_formatter.o
#SBATCH --error=./logs/PubMed_formatter.e
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=64
#SBATCH --partition=cpu64
#SBATCH --comment=carper
#SBATCH --exclusive
#SBATCH --export=ALL
# ===== END SLURM OPTIONS =====
source /fsx/home-erfan/miniconda3/bin/activate pilev2
cd /fsx/home-erfan/pilev2/pile/processing/fix_format
python fix_format_pubmed.py --data_dir /fsx/shared/pilev2/group1_filtered/PubMed --output_dir /fsx/shared/pilev2/group1_filtered/PubMed_formatted --do_sharding