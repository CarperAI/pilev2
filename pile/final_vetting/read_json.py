
import json
import boto3
import os
import typer
import tiktoken


def read_jsonl_from_s3(bucket_name, key=None, sample_size= None, start_index = 0) -> list:
    """
    Reads a JSONL file from an S3 bucket.

    Args:
    bucket_name (str): The name of the S3 bucket.
    key (str, optional): The key of the JSONL file in the S3 bucket. 
        If None, it will try to read the key from the environment variable S3_KEY.

    Returns:
    A list of Python objects representing the JSONL file.
    """
    typer.echo(f"Reading JSONL file from S3 bucket {bucket_name}...")
    s3 = boto3.resource('s3')
    # if key is None:
    #     key = os.environ.get('S3_KEY')
    bucket_name, dir = bucket_name.split("/", 1)

    response = s3.Object(bucket_name, dir).get()
    content = response['Body'].read().decode('utf-8')
    lines = content.strip().split('\n')
    if sample_size:
        lines = lines[start_index: start_index + sample_size]
    return [json.loads(line) for line in lines]

def decode_sequence(sequence):
    """
    Decodes a sequence of integers into a string.

    Args:
    sequence (list): A list of integers representing a sequence of characters.

    Returns:
    A string representing the decoded sequence.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(sequence['input_ids'])



# create main function
if __name__ == "__main__":
    """
    bucket_path is a required argument. It should be the name and path of the S3 bucket.
    key is an optional argument. If not provided, it will try to read the key from the environment variable S3_KEY.
    """
    def run(file1_path: str = typer.Argument(..., help="The name and path of the first file in the S3 bucket."),
            file2_path: str = typer.Option(None, help="The name and path of the second file in the S3 bucket."),
            key: str = typer.Option(None, help="The key of the JSONL file in the S3 bucket. If None, it will try to read the key from the environment variable S3_KEY."),
            sample_size: int = typer.Option(None, help="The number of lines to read from the JSONL file. If None, it will read the entire file."),
            start_index: int = typer.Option(0, help="The index of the first line to read from the JSONL file.")):
        
        
        """
        Reads a JSONL file from an S3 bucket.
        """
        # remove s3:// from bucket_path
        file1_path = file1_path.replace("s3://", "")
        if file2_path is not None:
            file2_path = file2_path.replace("s3://", "")
            file2_contents = [decode_sequence(seq) for seq in read_jsonl_from_s3(file2_path, key, sample_size=sample_size, start_index=start_index)]
        else:
            file2_contents = None

        file1_contents = [decode_sequence(seq) for seq in read_jsonl_from_s3(file1_path, key, sample_size=sample_size, start_index=start_index)]

        exact_matches = 0
        if file2_contents is not None:
            for f1_line, f2_line in zip(file1_contents, file2_contents):
                i = 0
                # compare the two lines if equal then print one and say they match if they don't print both and say they don't match

                if f1_line == f2_line:
                    print(f"Files match on line {i}")
                    print(decode_sequence(f1_line))
                    exact_matches += 1
                else:
                    print(f"Files do not match on line {i}")
                    print(f"File 1: {f1_line} \t File 2: {f2_line}")
                i += 1
            print(f"Exact matches: {exact_matches}")
        else:
            for f1_line in file1_contents:
                print(f1_line)

        

    
    typer.run(run)
