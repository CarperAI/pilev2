import argparse

import numpy as np

from datasets import disable_caching, load_from_disk, load_dataset
from pathlib import Path
from squeakily.core import Pipeline
from squeakily.clean import fix_utf8_encoding, replace_ip
from squeakily.filter import check_compression_ratio
from functools import partial
disable_caching()

replace_ip_p = partial(replace_ip, dummy2="::")
replace_ip_p.__name__ = "replace_ip_p"
# Parse the arguments

"""
Compression filtering of a dataset and ipv6 format fix, fixes utf8 encoding, and shards the data.
To be run on all datasets.
"""
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
    default=64,
    help="The number of processes to use for the filtering.",
)
parser.add_argument(
    "--num_files_per_shard",
    type=int,
    default=30_000,
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
num_proc = args.num_proc
data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
# list all parquet files in data_dir
data_files = [str(f) for f in list(data_dir.glob("*.parquet"))]
ds = load_dataset("parquet", data_files = {"train" : data_files})
ds = ds["train"]

if data_dir.name.startswith("the_stack_filtered"):
    columns = ["content"]
else:
    columns = ["text"]
# add to ds the following columns:
# - check_compression_ratio_criteria
# - fix_utf8_encoding_criteria
datasources = [
    {
        "dataset": ds ,
        "name": data_dir.name,
        "columns": columns,
        "filters": [check_compression_ratio],
        "cleaners": [fix_utf8_encoding, replace_ip_p],
    }
]
pipeline = Pipeline(datasources)
pipeline.run(dry_run=True, num_proc=num_proc)
new_ds = pipeline.datasources[0]["dataset"]
print(pipeline.datasources)
print(f"New dataset: {new_ds}")
start_size = len(new_ds)
compression_ratios = new_ds["check_compression_ratio_criteria"]
min_compression_ratio = np.quantile(compression_ratios, args.min_percentile)
new_ds = new_ds.filter(
    lambda x: x["check_compression_ratio_criteria"] > min_compression_ratio,
    batched=True,
    num_proc=num_proc,
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
