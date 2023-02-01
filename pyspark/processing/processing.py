import argparse

from functools import partial
from helpers import get_file_paths
from pyspark.sql.functions import lit, udf
from pyspark.sql.types import StringType, FloatType
from spark_session_builder import build_spark_session
from squeakily.clean import fix_utf8_encoding
from squeakily.filter import check_compression_ratio

# Setup the UDFs
udf_fix_utf8_encoding = udf(fix_utf8_encoding, StringType())
check_compression_ratio_p = partial(check_compression_ratio, dry_run=True)
check_compression_ratio_p.name = "check_compression_ratio"
udf_check_compression_ratio = udf(check_compression_ratio_p, FloatType())

parser = argparse.ArgumentParser()
parser.add_argument(
    "--main_addr",
    type=str,
    required=True,
    help="The address of the main node.",
)
parser.add_argument(
    "--bucket_name",
    type=str,
    required=True,
    help="The name of the bucket.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory where the data is stored.",
)
parser.add_argument(
    "--extension",
    type=str,
    required=True,
    help="The extension of the files to be processed.",
)
parser.add_argument(
    "--num_cpus_per_node",
    type=int,
    default=32,
    help="The number of CPUs per node.",
)
parser.add_argument(
    "--mem_per_node",
    type=int,
    default=48,
    help="The amount of memory per node in GB.",
)

args = parser.parse_args()

spark = build_spark_session(f"spark://{args.main_addr}:7077", args.num_cpus_per_node, args.mem_per_node)
paths = get_file_paths(args.bucket_name, args.data_dir, args.extension)
print(paths.keys())

# read in the data and add a column for the data source
dfs = []
for data_source, files in paths.items():
    print(f"Processing {data_source}...")
    df = spark.read.parquet(*files)
    df = df.withColumn("data_source", lit(data_source))
    dfs.append(df)

    # # clean the data
    # df = fix_utf8_encoding(df)

    # # filter the data
    # df = check_compression_ratio(df)

    # # write the data to disk
    # df.write.parquet(f"s3a://{args.bucket_name}/{args.data_dir}/{data_source}/cleaned.parquet")

# combine the data
df = dfs[0]
for i in range(1, len(dfs)):
    df = df.union(dfs[i])

# Fix the UTF-8 encoding of the text column
df = df.withColumn("text", udf_fix_utf8_encoding("text"))

# Calculate the compression ratio of the text column
df = df.withColumn("compression_ratio", udf_check_compression_ratio("text"))

# Filter the data
quantile = df.approxQuantile("compression_ratio", [0.01], 0.0)[0]
df = df.filter(df.compression_ratio > quantile)

# Write the data to disk per data source
for data_source in paths.keys():
    df.filter(df.data_source == data_source).write.parquet(f"s3a://{args.bucket_name}/{args.data_dir}/{data_source}/cleaned.parquet")
