import argparse

import numpy as np

from datasets import load_from_disk, load_dataset
from pathlib import Path
from squeakily.core import Pipeline
from squeakily.clean import fix_utf8_encoding
from squeakily.filter import check_compression_ratio

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory where the data is stored.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="The directory where the output should be stored.",
)
parser.add_argument(
    "--min_percentile",
    type=float,
    default=0.01,
    help="The minimum percentile to use for the threshold.",
)
parser.add_argument(
    "--num_proc",
    type=int,
    default=128,
    help="The number of processes to use for the filtering.",
)
parser.add_argument(
    "--num_files_per_shard",
    type=int,
    default=10_000,
    help="The number of files per shard.",
)
parser.add_argument(
    "--do_sharding",
    action="store_true",
    help="Whether to shard the data.",
)

parser.add_argument(
    "--save_format",
    type=str,
    default="parquet",
    help="The format to save the data in. Options are 'parquet', 'JSONL', and 'arrow'.",
)

args = parser.parse_args()
data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
ds = load_dataset("parquet", data_files = str(data_dir))
# add to ds the following columns:
# - check_compression_ratio_criteria
# - fix_utf8_encoding_criteria
datasources = [
    {
        "dataset": ds ,
        "name": data_dir.name,
        "columns": ["text"],
        "filters": [check_compression_ratio],
        "cleaners": [fix_utf8_encoding],
    }
]
pipeline = Pipeline(datasources)
pipeline.run(dry_run=True, num_proc=96)
new_ds = pipeline.datasources[0]["dataset"]['train']
start_size = len(new_ds)
compression_ratios = new_ds["check_compression_ratio_criteria"]
min_compression_ratio = np.quantile(compression_ratios, args.min_percentile)
new_ds = new_ds.filter(
    lambda x: x["check_compression_ratio_criteria"] > min_compression_ratio,
    batched=True,
    num_proc=32,
)
num_shards = 0
if args.do_sharding:
    num_shards = len(new_ds) // args.num_files_per_shard
if num_shards == 0:
    num_shards = 1
ds_shards = [new_ds.shard(num_shards, i, contiguous=True) for i in range(num_shards)]
# get file name from data_dir
file_name = data_dir.name
# remove extension
file_name = file_name.split(".")[0]
for i, shard in enumerate(ds_shards):
    if args.save_format == "parquet":
        path = output_dir / f"{file_name}_shard_{i}.parquet" if i > 0 else output_dir / f"{file_name}.parquet"
        shard.to_parquet(path)
    elif args.save_format == "arrow":
        path = output_dir / f"{file_name}_shard_{i}.arrow" if i > 0 else output_dir / f"{file_name}.arrow"
        shard.save_to_disk(path)
    elif args.save_format == "JSONL":
        #TODO: Use lm_dataformat to save the data.
        path = output_dir / f"{file_name}_shard_{i}.jsonl.zst" if i > 0 else output_dir / f"{file_name}.jsonl.zst"
        shard.to_json(
            path,
            lines=True,
            orient="records",
        )