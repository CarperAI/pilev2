#!/bin/bash
#
# get environment variables
GLOBAL_RANK=$SLURM_PROCID
CPUS=`grep -c ^processor /proc/cpuinfo`
MEM=$((`grep MemTotal /proc/meminfo | awk '{print $2}'`/1000)) # seems to be in MB
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
LOCAL_IP=$(hostname -I | awk '{print $1}')

# setup the master node
if [ $GLOBAL_RANK == 0 ]
then
    # print out some info
    echo -e "MASTER ADDR: $MASTER_ADDR\tGLOBAL RANK: $GLOBAL_RANK\tCPUS PER TASK: $CPUS\tMEM PER NODE: $MEM"

    # then start the spark master node in the background
    ./spark-3.3.1-bin-hadoop3/sbin/start-master.sh -p 7077 -h $LOCAL_IP
   
fi

sleep 10

# then start the spark worker node in the background
MEM_IN_GB=$(($MEM / 1000))
# concat a "G" to the end of the memory string
MEM_IN_GB="$MEM_IN_GB"G
echo "MEM IN GB: $MEM_IN_GB"

./spark-3.3.1-bin-hadoop3/sbin/start-worker.sh -c $CPUS -m $MEM_IN_GB "spark://$MASTER_ADDR:7077"
echo "Hello from worker $GLOBAL_RANK"

sleep 10

if [ $GLOBAL_RANK == 0 ]
then
    # then start some script
    echo "hi"
fi

sleep 1000000
