import argparse
import os

from datasets import load_dataset, disable_caching
from pathlib import Path
from decontamination.core import BenchmarkCleaner
disable_caching()

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
args = parser.parse_args()
data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)

benchmark_names = ["bigscience/P3", "codeparrot/apps", "wino_bias", "openai_humaneval", "mbpp", "ncoop57/mmmlu", "lambada"]
bench_cleaner = BenchmarkCleaner(benchmark_names, output_dir / "benchmarks", threshold=0.85, num_perm=256, num_workers=48)
parquets = [str(par) for par in data_dir.glob("*.parquet")]
try:
    ds = load_dataset("parquet", data_files=parquets, split="train", num_proc=116)
except Exception as e:
    print(e)
    print("Error")
    exit(1)

ds = bench_cleaner.clean(ds, "text")
ds.save_to_disk(output_dir / data_dir.name)