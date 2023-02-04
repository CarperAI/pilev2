import argparse
import re
import yaml

from functools import partial
from datasets import disable_caching, load_from_disk, load_dataset
from pathlib import Path
from squeakily.clean import replace_ip
from squeakily.core import Pipeline
disable_caching()


replace_ip_p = partial(replace_ip, dummy2="::")
replace_ip_p.__name__ = "replace_ip_p"

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
# add sharding options
parser.add_argument(
    "--do_shard",
    action="store_true",
    help="Whether to shard the data.",
)
parser.add_argument(
    "--num_files_per_shard",
    type=int,
    default=30_000,
    help="The number of files per shard.",
)

args = parser.parse_args()
data_dir = Path(args.data_dir)
data_files = [str(p) for p in data_dir.glob("*.parquet") if p.is_file()]

# create the output directory if it does not exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# if the data_dir contains the_stack_filtered then change the columns to "content"
if data_dir.name.startswith("the_stack_filtered"):
    columns = ["content"]
else:
    columns = ["text"]

datasources = [
    {
        "dataset": load_dataset("parquet", data_files={'train': data_files})['train'],
        "name": data_dir.name,
        "columns": columns,
        "filters": [],
        "cleaners": [replace_ip_p],
    }
]

pipeline = Pipeline(datasources)
pipeline.run()

print(pipeline.datasources)


    

# save the resulting datasets
for ds in pipeline.datasources:
    # make directories for each datasource
    (output_dir / ds["name"]).mkdir(parents=True, exist_ok=True)
    # save the dataset with sharding
    if args.do_shard:
        num_shards = 0
        if args.do_sharding:
            num_shards = len(ds['dataset']) // args.num_files_per_shard
        if num_shards == 0:
            num_shards = 1
        ds_shards = [ds['dataset'].shard(num_shards, i, contiguous=True) for i in range(num_shards)]
        # get file name from data_dir
        file_name = data_dir.name
        # save the shards
        for i, shard in enumerate(ds_shards):
            path = output_dir / f"{file_name}_shard_{i}.parquet" if i > 0 else output_dir / f"{file_name}.parquet"
            shard.to_parquet(path)