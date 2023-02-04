#! /bin/bash

python subsample_data.py \
    --main_addr cpu16-dy-r6i-4xlarge-4 \
    --bucket_name s-eai-neox \
    --data_dir data/codepile/group2/ \
    --extension .parquet \
    --num_cpus_per_node 4 \
    --mem_per_node 4 \
    --percent 0.25