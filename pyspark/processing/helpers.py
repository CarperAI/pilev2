import boto3

from collections import defaultdict

def get_file_paths(bucket, prefix, extension):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket)
    result = defaultdict(list)
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith(extension):
            result[obj.key.split("/")[-2]].append(obj.key)
    return result