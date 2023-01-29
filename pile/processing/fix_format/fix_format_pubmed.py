import argparse
import re
import yaml

from datasets import disable_caching, load_from_disk, load_dataset
from pathlib import Path
disable_caching()

def get_body_text(text):
  lines = text.split("\n")
  body_contents = []
  in_body = False
  start, end = 0, 0
  for idx, line in enumerate(lines):
    if "==== Body" in line:
      in_body = True
      start = idx
      continue
    elif "==== " in line and in_body:
      end = idx
      break
    if in_body:
      body_contents.append(line)

  return "\n".join(body_contents), (start, end)

def reformatter(example):
  text, (start, end) = get_body_text(example["text"])
  example["text"] = text
  return example


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
    "--num_proc",
    type=int,
    default=56,
    help="The number of processes to use.",
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

args = parser.parse_args()

# create the output directory if it does not exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

data_dir = Path(args.data_dir)
# pubmed_ds = load_from_disk(data_dir)
# load from parquet
num_proc=args.num_proc
pubmed_ds = load_dataset("parquet", data_dir=data_dir)
pubmed_ds = pubmed_ds["train"]
pubmed_ds = pubmed_ds.map(reformatter
                          , num_proc=num_proc)
print(pubmed_ds[0]["text"])

num_shards = 0
if args.do_sharding:
    num_shards = len(pubmed_ds) // args.num_files_per_shard
if num_shards == 0:
    num_shards = 1
ds_shards = [pubmed_ds.shard(num_shards, i, contiguous=True) for i in range(num_shards)]
# get file name from data_dir
file_name = data_dir.name
# remove extension
file_name = file_name.split(".")[0]
for i, shard in enumerate(ds_shards):
      path = output_dir / f"{file_name}_shard_{i}.parquet" if i > 0 else output_dir / f"{file_name}.parquet"
      shard.to_parquet(path)


# python fix_format_pubmed.py --data_dir /work/pilev2/pile/processing/fix_format/pile-v2-eda/local_dedup/PubMed_ver2 --output_dir /work/pilev2/pile/processing/fix_format/pile-v2-eda/reformatted/PubMed_ver2

# /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/local_dedup/arXiv_ver2
# /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/reformated/arXiv_ver2