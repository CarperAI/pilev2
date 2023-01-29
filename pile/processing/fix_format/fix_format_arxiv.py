import argparse
import re
import yaml

from datasets import disable_caching, load_from_disk, load_dataset
from pathlib import Path
disable_caching()

def get_abs_title(text):
  # Remove any non-ascii characters and x0007
  text = re.sub(r'\x07',r'', text)
  text = re.sub(r'[^\x00-\x7f]',r'', text)
  # remove single quotes
  # text = re.sub(r"'",r'', text)
  lines = text.split("\n")
  yaml_contents = []
  in_yaml = False
  start, end = 0, 0
  for idx, line in enumerate(lines):
    if line == "---":
      if in_yaml:
        end = idx
        break
      else:
        in_yaml = True
        start = idx
        continue
    if in_yaml:
      yaml_contents.append(line)
  
  abstract = ""
  title = ""
  for idx, line in enumerate(yaml_contents):
    if line.startswith("title:"):
      title = line.replace("title:", "")
    if line.startswith("abstract:"):
      abstract = line.replace("abstract:", "")

  # trim the title and abstract
  title = title.strip()
  abstract = abstract.strip()

  # remove the beginning and ending quotes
  if len(title) > 0:
    if title[0] == "'":
      title = title[1:]
    if title[-1] == "'":
      title = title[:-1]
  if len(abstract) > 0:
    if abstract[0] == "'":
      abstract = abstract[1:]
    if abstract[-1] == "'":
      abstract = abstract[:-1]
  return title, abstract, (start, end)

def reformatter(example):
  title, abstract, (start, end) = get_abs_title(example["text"])

  # remove the yaml contents from the text
  lines = example["text"].split("\n")
  lines = lines[:start] + lines[end+1:]
  # add the title and abstract
  lines = [title, "", abstract] + lines

  example["text"] = "\n".join(lines)
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
    "--num_proc",
    type=int,
    default=8,
    help="The number of processes to use.",
)
args = parser.parse_args()

# create the output directory if it does not exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

num_proc = args.num_proc
data_dir = Path(args.data_dir)
# load from parquet
arxiv_ds = load_dataset("parquet", data_dir=data_dir)
arxiv_ds = arxiv_ds["train"]
# arxiv_ds = load_from_disk(data_dir)
arxiv_ds = arxiv_ds.map(reformatter, num_proc=num_proc)
print(arxiv_ds[0]["text"])
num_shards = 0
if args.do_sharding:
    num_shards = len(arxiv_ds) // args.num_files_per_shard
if num_shards == 0:
    num_shards = 1
ds_shards = [arxiv_ds.shard(num_shards, i, contiguous=True) for i in range(num_shards)]
# get file name from data_dir
file_name = data_dir.name
# remove extension
file_name = file_name.split(".")[0]
for i, shard in enumerate(ds_shards):
      path = output_dir / f"{file_name}_shard_{i}.parquet" if i > 0 else output_dir / f"{file_name}.parquet"
      shard.to_parquet(path)
# arxiv_ds.save_to_disk(output_dir)

# python fix_format_arxiv.py --data_dir /work/pilev2/pile/processing/fix_format/pile-v2-eda/local_dedup/arXiv_ver2 --output_dir /work/pilev2/pile/processing/fix_format/pile-v2-eda/reformated/arXiv_ver2
# /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/local_dedup/arXiv_ver2
# /fsx/home-nathan/work/pilev2/pile/processing/fix_format/pile-v2-eda/reformated/arXiv_ver2