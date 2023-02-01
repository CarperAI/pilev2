import argparse
import re
import yaml

from functools import partial
from datasets import disable_caching, load_from_disk
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
args = parser.parse_args()
data_dir = Path(args.data_dir)
data_dirs = [Path(d) for d in data_dir.iterdir() if d.is_dir()]

# create the output directory if it does not exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

datasources = [
    {
        "dataset": load_from_disk(d),
        "name": d.name,
        "columns": ["text"],
        "filters": [],
        "cleaners": [replace_ip_p],
    }
    for d in data_dirs
]

pipeline = Pipeline(datasources)
pipeline.run()

print(pipeline.datasources)

# Save the resulting datasets
for d, ds in zip(data_dirs, pipeline.datasources):
    ds["dataset"].save_to_disk(output_dir / d.name)