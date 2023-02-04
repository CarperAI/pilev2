import argparse
import boto3

from collections import defaultdict

def default_arguments():
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
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the output should be stored.",
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

    return parser

def get_file_paths(bucket_name, prefix, extension):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    result = defaultdict(list)
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith(extension):
            result[obj.key.split("/")[-2]].append(f"s3a://{bucket_name}/{obj.key}")
    return result