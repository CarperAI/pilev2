from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, MinHashLSH
from pyspark.sql.functions import col
from spark_session_builder import build_spark_session
from pyspark.sql.functions import length
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
subfolder_paths = set()
for obj in objects:

    if obj["Key"].endswith(".json.gz"):
        folder = "/".join(obj["Key"].split("/")[:-1])
        if folder not in json_gz_paths:
            json_gz_paths [folder] = []
        json_gz_paths [folder].append(obj["Key"])
        subfolder_paths.add(folder)


# create a PySpark DataFrame for each folder that contains JSON files compressed with gzip
for path in subfolder_paths:
    print(f"Processing folder: {path}") 
    json_files = [f"s3a://{bucket}/{p}" for p in json_gz_paths[path]]
    print(f"Processing files: {json_files}")
    # read all json.gz files in the directory as a single DataFrame
    json_df = spark.read.json(*json_files)
    # select only the input_ids attribute and compute its length
    length_df = json_df.select(col("input_ids").alias("input_ids_length"))
    length_df = length_df.withColumn("input_ids_length", length_df["input_ids_length"].apply(len))


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
    subset_stats[display_path] = {"mean": subset_token_mean, "median": subset_token_median, "std": subset_token_std, "min": subset_token_min, "max": subset_token_max, "total_tokens": subset_token_size}


# create a PySpark DataFrame from the dictionary of lengths
stats_df = spark.createDataFrame([(k, v) for k, v in subset_stats.items()], ["subset", "mean", "median", "std", "min", "max", "total_tokens"])

# compute the ratio of the length of each subfolder to the average length
# round to 2 decimal places
stats_df = stats_df.withColumn("pilev2_ratio", round(stats_df["total_tokens"] / (total_length)), 2)

# output directory setup
output_dir = args.output_dir
if output_dir is None:
    output_dir = os.path.join(os.getcwd(), "pilev2_stats")
Path(output_dir).mkdir(parents=True, exist_ok=True)

# save the DataFrame as a CSV file inside the output directory
length_df.toPandas().to_csv(os.path.join(output_dir, "subset_ratios.csv"), index=False)

# save the DataFrame as a JSON file and then text file
length_df.toPandas().to_json(os.path.join(output_dir, "subset_ratios.json"), orient="records")

# save the DataFrame as a table in a text file using tabulate
from tabulate import tabulate
with open(os.path.join(output_dir, "subset_ratios.txt"), "w") as f:
    f.write(tabulate(length_df.toPandas(), headers="keys", tablefmt="psql"))






