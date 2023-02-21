#! /bin/bash
# job name is the basename of the dataset and file
#SBATCH --job-name=rn_stats
#SBATCH --output=./logs/rn_stats.o
#SBATCH --error=./logs/rn_stats.e
#SBATCH --exclusive
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu16
#SBATCH --comment=carper
#SBATCH --export=ALL
# ===== END SLURM OPTIONS =====


# PySpark computing token statistics for the pilev2 dataset

S3_PATH=s3a://stability-llm/pilev2_tokenized
OUTPUT_PATH=/fsx/shared/pilev2/pilev2_stats

#TODO: Change me
spark_path=/fsx/home-erfan/pilev2/pyspark

# Set up logs directories
mkdir -p $spark_path/logs
mkdir -p $spark_path/processing/logs

# set up the spark cluster
source /fsx/home-erfan/miniconda3/bin/activate pilev2
cd $spark_path

# run the spark cluster and record the job id
job_id=$(sbatch spark_on_slurm.sh | cut -d ' ' -f 4)
echo $job_id

# Check if the cluster is up and running
# If not, wait 30 seconds and try again
while [ ! -f $spark_path/logs/hostnames.txt ]; do
    sleep 30
done

# get the node name
hostnames=$(cat $spark_path/logs/hostnames.txt)
min_node_number=1000
# get the first hostname
# loop through the hostnames and get the one with the lowest number
for hostname in $hostnames; do
    # get the last element of the hostname
    # this is the node number
    node_number=$(echo $hostname | cut -d '-' -f 5)
    # if min_node_number is default of 1000, set it to the first node number
    if [ $min_node_number -eq 1000 ]; then
        min_node_number=$node_number
        min_hostname=$hostname

    elif [ $node_number -lt $min_node_number ]; then
        min_node_number=$node_number
        min_hostname=$hostname
    fi
done

node_name=$min_hostname

# print connecting to node
echo "Connecting to node $node_name"

cd $spark_path/processing
python -u compute_subset_stats.py --data_dir $S3_PATH --output_dir $OUTPUT_PATH --node $node_name --num_cores 16 --memory 64

# kill the spark cluster
scancel $job_id

# remove the hostnames file
rm $spark_path/logs/hostnames.txt