#!/bin/bash
#SBATCH --partition=cpu32
#SBATCH --job-name=pyspark_pilev2
#SBATCH --nodes 3
#SBATCH --cpus-per-task=32
#SBATCH --mem=0 # 0 means use all available memory (in MB)
#SBATCH --output=%x_%j.out
#SBATCH --comment carper
#SBATCH --exclusive
#SBATCH --export=ALL
# ===== END SLURM OPTIONS =====
source /fsx/home-erfan/miniconda3/bin/activate pilev2
cd /fsx/home-erfan/pilev2/pyspark
# Write the hostnames of the nodes to a file
srun --comment carper hostname > ./logs/hostnames.txt
# Start the Spark cluster
srun --comment carper bash worker_spark_on_slurm.sh
