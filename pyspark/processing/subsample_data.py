import argparse
import boto3

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from helpers import default_arguments, get_file_paths
from spark_session_builder import build_spark_session

from datasets import load_dataset
from pathlib import Path


parser = default_arguments()
parser.add_argument(
    "--percent",
    type=float,
    default=0.25,
    help="The output directory.",
)
args = parser.parse_args()

paths_dict = get_file_paths(args.bucket_name, args.data_dir, args.extension)
spark = build_spark_session(f"spark://{args.main_addr}:7077", args.num_cpus_per_node, args.mem_per_node)

for data_source, paths in paths_dict.items():
    print(f"Processing {data_source}...")
    df = spark.read.parquet(*paths)
    print(f"Number of rows: {df.count()}")
    sample_df = df.sample(withReplacement=False, fraction=args.percent)
    print(f"Number of rows in sample: {sample_df.count()}")
    sample_df.write.parquet(f"s3a://{args.bucket_name}/{args.data_dir}/{data_source}_subsample/{args.percent}.parquet")
    break