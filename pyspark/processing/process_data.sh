#! /bin/bash

python processing.py \
    --main_addr cpu16-dy-r6i-4xlarge-4 \
    --bucket_name s-eai-neox \
    --data_dir data/pilev2/raw_data/ \
    --output_dir data/pilev2/processed_data/ \
    --extension jsonl.zst \
    --num_cpus_per_node 4 \
    --mem_per_node 4