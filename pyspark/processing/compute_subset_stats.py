from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, MinHashLSH
from pyspark.sql.functions import col
from spark_session_builder import build_spark_session
import tiktoken
import os
import argparse
from pathlib import Path
import json
import os
import boto3
import gzip
import io
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import pandas as pd
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType
from pyspark.sql.functions import from_json


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
    required=False,
    help="The path where the output should be stored.",
)

parser.add_argument(
    "--node",
    type=str,
    required=False,
    default="cpu32-dy-m6i-8xlarge-1",
    help="The node to connect to.",
)

# get num_cores and memory from args
parser.add_argument(
    "--num_cores",
    type=int,
    required=False,
    default=16,
    help="The number of cores to use.",
)

parser.add_argument(
    "--memory",
    type=int,
    required=False,
    default=32,
    help="The amount of memory in GB to use.",
)


args = parser.parse_args()

spark = build_spark_session(f"spark://{args.node}:7077", args.num_cores, args.memory)
    
root = args.data_dir
bucket = root.replace("s3a://", "").split("/")[0]
prefix = "/".join(root.replace("s3a://", "").split("/")[1:])
# create an S3 client
s3 = boto3.client("s3")
# initialize a variable to hold the total length of input_ids across all subfolders
total_length = 0
# initialize a variable to hold the length of input_ids per subfolder
subset_lengths = {}
# initialize a variable to hold the statistics.
subset_stats = {}

# list all objects in the S3 path
objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)["Contents"]

# initialize a set to hold the paths of the folders that contain JSON files compressed with gzip
json_gz_paths = {}

# loop through each object in the S3 path and find the paths of the folders that contain JSON files compressed with gzip
# subfolder_paths = set()
# for obj in objects:

#     if obj["Key"].endswith(".json.gz"):
#         folder = "/".join(obj["Key"].split("/")[:-1])
#         if folder not in json_gz_paths:
#             json_gz_paths [folder] = []
#         json_gz_paths [folder].append(obj["Key"])
#         subfolder_paths.add(folder)

subfolder_paths = [
    'the_stack',
    'ai4code_nbs',
    'amps',
    'arxiv',
    'cc2',
    'code_reddit_dialog',
    'code_reddit_posts',
    'competitive_programming',
    'devdocs',
    'discourse',
    'dm_math',
    'enwiki',
    'euro_parl',
    'free_law',
    'github_diffs',
    'github_issues',
    'gutenberg',
    'open_subtitles',
    'other_wiki',
    'pile_of_law',
    'pubmed',
    'reddit_dialogs',
    'reddit_posts',
    's2orc',
    'soda',
    'stack_exchange',
    'ted_talks',
    'ubuntu_irc',
    'uspto'
]
subfolder_paths = [f'pilev2_tokenized/{p}' for p in subfolder_paths]
subfolder_paths = set(subfolder_paths)
schema = StructType([
    StructField("input_ids", ArrayType(IntegerType()), True),
    StructField("attention_mask", ArrayType(IntegerType()), True)
])
s3 = boto3.resource('s3')
def get_all_files(prefix):
    bucket_name = "stability-llm"
    bucket = s3.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=prefix)
    files = [obj.key for obj in objects if obj.key != prefix]  # exclude the folder itself from the results
    return [f"s3a://{bucket_name}/{f}" for f in files if f.endswith('.json.gz') or f.endswith('.txt.gz')]


# create a PySpark DataFrame for each folder that contains JSON files compressed with gzip
for path in list(subfolder_paths):
    print(f"Processing folder: {path}") 
    # json_files = [f"s3a://{bucket}/{p}" for p in json_gz_paths[path]]
    json_files = get_all_files(path)
    print(f"Processing {len(json_files)} files...")
    # read all json.gz files in the directory as a single DataFrame
    if json_files[0].endswith('.txt.gz'):
        json_df = spark.read.text(json_files)
        json_df = json_df.select(from_json(json_df.value, schema).alias("data")).select("data.*")
    else:
        json_df = spark.read.json(json_files)
    # select only the input_ids attribute and compute its length
    length_df = json_df.select(col("input_ids").alias("input_ids_length"))
    # apply the python len function to each element of the input_ids attribute
    length_df = length_df.withColumn("input_ids_length", udf(lambda x: len(x), IntegerType())("input_ids_length"))

    # compute number of documents in the directory
    subset_num_docs = length_df.count()
    # compute the total length across all files in the directory
    subset_token_size = length_df.agg({"input_ids_length": "sum"}).collect()[0][0]
    # compute mean, median, and standard deviation of the length of input_ids and store in a dictionary
    subset_token_mean = length_df.agg({"input_ids_length": "mean"}).collect()[0][0]
    subset_token_median = length_df.approxQuantile("input_ids_length", [0.5], 0.25)[0]
    subset_token_std = length_df.agg({"input_ids_length": "stddev"}).collect()[0][0]
    # compute min and max of the length of input_ids and store in a dictionary
    subset_token_min = length_df.agg({"input_ids_length": "min"}).collect()[0][0]
    subset_token_max = length_df.agg({"input_ids_length": "max"}).collect()[0][0]
    # add the length of the current directory to the total length
    total_length += subset_token_size
    # add the length of the current directory to the dictionary of lengths
    # display_path is formatted path to include only the name of the subfolder
    display_path = path.split("/")[-1]
    subset_lengths[display_path] = subset_token_size
    # add the statistics of the current directory to the dictionary of statistics
    subset_stats[display_path] = {"mean": subset_token_mean, "median": subset_token_median, "std": subset_token_std, "min": subset_token_min, "max": subset_token_max, "total_tokens": subset_token_size, "num_docs": subset_num_docs}


# create a PySpark DataFrame from the dictionary of subset stats
stats_df = spark.createDataFrame(pd.DataFrame(subset_stats).T.reset_index().rename(columns={"index": "subset"}))

# compute the ratio of the length of each subfolder to the average length
# round to 2 decimal places
stats_df = stats_df.withColumn("pilev2_ratio", stats_df["total_tokens"] / (total_length))

# output directory setup
output_dir = args.output_dir
if output_dir is None:
    output_dir = os.path.join(os.getcwd(), "pilev2_stats")
Path(output_dir).mkdir(parents=True, exist_ok=True)

# save the DataFrame as a CSV file inside the output directory
stats_df.toPandas().to_csv(os.path.join(output_dir, "subset_ratios.csv"), index=False)

# save the DataFrame as a JSON file and then text file
stats_df.toPandas().to_json(os.path.join(output_dir, "subset_ratios.json"), orient="records")

# save the DataFrame as a table in a text file using tabulate
from tabulate import tabulate
with open(os.path.join(output_dir, "subset_ratios.txt"), "w") as f:
    f.write(tabulate(stats_df.toPandas(), headers="keys", tablefmt="psql"))


# Write total token count to a text file
with open(os.path.join(output_dir, "total_tokens.txt"), "w") as f:
    f.write(str(total_length))




