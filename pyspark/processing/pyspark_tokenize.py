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

parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    
    help="The name of the dataset.",
)

# get the cluster job id from args
parser.add_argument(
    "--cluster_job_id",
    type=str,
    required=False,
    default="",
    help="The cluster job id.",
)



args = parser.parse_args()

spark = build_spark_session(f"spark://{args.node}:7077", args.num_cores, args.memory)

def tokenize(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    result = tokenizer.encode(text, disallowed_special=())
    return json.dumps({"input_ids": result, "attention_mask": [1] * len(result)})

data_dir = Path(args.data_dir)
files = [str(f) for f in list(data_dir.glob("*.parquet"))]

print(files)

if args.output_dir is None:
    root = f"/fsx/shared/pilev2/group3_filtered_decontam/the_stack_jsonl/{args.data_dir.split('/')[-1]}"
else:
    root = args.output_dir

data = spark.read.parquet(*files)
find_tokenize = udf(tokenize)

data = data.withColumn("tokenized", find_tokenize(data["text"]))
data = data.select("tokenized")

# print("Number of files: ", num_files)
# Write each partition to a separate JSONL file
max_file_size = 250 * 1024 * 1024 # 250 MB
(
    data.write
    .option("maxFileSize", max_file_size)
    .format("json")
    .mode("overwrite")
    .text(
        args.output_dir,
        compression="gzip",
        lineSep ="\n"
    )
)


# Rename all the files to .jsonl in the output directory
# This is because the spark jsonl writer adds a .txt.gz extension
# Handle this if output_dir is in s3 using the aws cli and os.system
if args.output_dir.startswith("s3a://"):
    output_dir = args.output_dir.replace("s3a://", "")
    # get all the files in the output directory
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(output_dir.split("/")[0])
    bucket_name = output_dir.split("/")[0]
    # get the path until the dataset name
    #TODO: This path thing does not work and is not general
    # Use something like this: https://stackoverflow.com/questions/71577584/python-boto3-s3-list-only-current-directory-file-ignoring-subdirectory-files
    dataset_name = '/'.join(output_dir.split("/")[:-1])
    files = [f"s3://{bucket_name}/{dataset_name}/{f.key}" for f in bucket.objects.filter(Prefix=output_dir.split("/")[-1])]
    print(files)
    for file in files:
        if file.endswith(".txt.gz"):
            print(f"aws s3 mv {file} {file.replace('.txt.gz', '.json.gz')}")
            os.system(f"aws s3 mv {file} {file.replace('.txt.gz', '.json.gz')}")
           
else:
    for file in os.listdir(args.output_dir):
        if file.endswith(".txt.gz"):
            os.rename(os.path.join(args.output_dir, file), os.path.join(args.output_dir, file.replace(".txt.gz", ".json.gz")))


# Run scancel on the job id to terminate the cluster
print(f"scancel {args.cluster_job_id}")
os.system(f"scancel {args.cluster_job_id}")