#! /bin/bash
# Depreciated not working properly.
# PySpark tokenization for group 2
# Set variables
DATA_PATH=/fsx/shared/pilev2/group3_filtered_decontam/group3_parquet
# DO NOT include a trailing slash in the output path
OUTPUT_PATH=s3a://stability-llm/jsonl_tiktoken_tests

#TODO: Change me
spark_path=/fsx/home-erfan/pilev2/pyspark

# Set up logs directories
mkdir -p $spark_path/logs
mkdir -p $spark_path/processing/logs

# Create a list all of the directories in $DATA_PATH
# datasets=$(ls $DATA_PATH)
datasets=("TED2020" "AI4Code_ver2")
cpus=16
mem=32
partition=cpu32
nodes=2
pyspark_server=./pyspark_server.slurm

# Loop through groups and submit slurm job for each dataset
for dataset in ${datasets[@]}; do   
    # list all files in $DATA_PATH/$dataset
    temp_dir=$DATA_PATH/$dataset
    basename_dataset=$(basename $dataset)
    basename_dataset=${basename_dataset%.*}
    echo $DATA_PATH/$dataset
    cat << HELP > $pyspark_server
#!/bin/bash
# job name is the basename of the dataset and file
#SBATCH --job-name=cls_$basename_dataset
#SBATCH --output=./logs/$basename_dataset.o
#SBATCH --error=./logs/$basename_dataset.e
#SBATCH --nodes=$nodes
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=$cpus
#SBATCH --partition=$partition
#SBATCH --exclusive
#SBATCH --comment=carper
#SBATCH --export=ALL
# ===== END SLURM OPTIONS =====
source /fsx/home-erfan/miniconda3/bin/activate pilev2
cd $spark_path
sleep 2
# Write the hostnames of the nodes to a file
srun --comment carper hostname > ./logs/$basename_dataset.hostnames.txt
# Start the Spark cluster
srun --comment carper bash worker_spark_on_slurm.sh
HELP
    # sbatch $pyspark_server 
    # save the job id from the output of sbatch
    job_id=$(sbatch $pyspark_server | cut -d ' ' -f 4)
    echo $job_id
    rm $pyspark_server
    # Check if the cluster is up and running
    # If not, wait 30 seconds and try again
    while [ ! -f $spark_path/logs/$basename_dataset.hostnames.txt ]; do
        sleep 30
    done
    # add job id to a file
    echo $job_id >> $spark_path/logs/$basename_dataset.jobid.txt
    # submit job to tokenize the dataset
    # read the hostnames from the file
    hostnames=$(cat $spark_path/logs/$basename_dataset.hostnames.txt)
    echo $spark_path/logs/$basename_dataset.hostnames.txt
    echo $hostnames
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

    echo $node_name
    # get the job id as a string
    cluster_job_id=$(cat $spark_path/logs/$basename_dataset.jobid.txt)
    # get the first element of the job id
    cluster_job_id=$(echo $cluster_job_id | cut -d ' ' -f 1)
    echo $cluster_job_id
    worker_submission=./worker.slurm
cat << HELP > $worker_submission
#!/bin/bash
# job name is the basename of the dataset and file
#SBATCH --job-name=rn_$basename_dataset
#SBATCH --output=./logs/rn_$basename_dataset.o
#SBATCH --error=./logs/rn_$basename_dataset.e
#SBATCH --exclusive
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu16
#SBATCH --comment=carper
#SBATCH --export=ALL
# ===== END SLURM OPTIONS =====
source /fsx/home-erfan/miniconda3/bin/activate pilev2
cd $spark_path/processing
python pyspark_tokenize.py \
    --data_dir $DATA_PATH/$dataset \
    --output_dir $OUTPUT_PATH/$dataset \
    --node $node_name \
    --num_cores $cpus \
    --memory $mem \
    --cluster_job_id $cluster_job_id \
    --dataset_name $basename_dataset
HELP
    sbatch $worker_submission
    rm $worker_submission
    # pause for 2 seconds.
    sleep 2
done
