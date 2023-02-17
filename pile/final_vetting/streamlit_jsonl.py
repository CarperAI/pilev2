import json
import boto3
import os
import streamlit as st
import tiktoken


def read_jsonl_from_s3(bucket_name, key=None, sample_size=None) -> list:
    """
    Reads a JSONL file from an S3 bucket.

    Args:
    bucket_name (str): The name of the S3 bucket.
    key (str, optional): The key of the JSONL file in the S3 bucket.
        If None, it will try to read the key from the environment variable S3_KEY.

    Returns:
    A list of Python objects representing the JSONL file.
    """
    st.write(f"Reading JSONL file from S3 bucket {bucket_name}...")
    s3 = boto3.resource('s3')
    bucket_name, dir = bucket_name.split("/", 1)

    response = s3.Object(bucket_name, dir).get()
    content = response['Body'].read().decode('utf-8')
    lines = content.strip().split('\n')
    if sample_size:
        lines = lines[:sample_size]
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


def main():
    """
    Launches a Streamlit app that displays the contents of two selected files.
    """
    st.title("JSONL Viewer")
    st.write("Enter the file paths of two JSONL files stored in an S3 bucket below.")
    
    # Get file paths from user input
    bucket_path1 = st.text_input("Bucket Path 1")
    key1 = st.text_input("Key 1 (optional)")
    bucket_path2 = st.text_input("Bucket Path 2")
    key2 = st.text_input("Key 2 (optional)")
    
    # Read file contents from S3 bucket
    if bucket_path1 and bucket_path2:
        file_contents1 = read_jsonl_from_s3(bucket_path1, key=key1)
        file_contents2 = read_jsonl_from_s3(bucket_path2, key=key2)

        # Display file contents in two columns
        col1, col2 = st.beta_columns(2)
        with col1:
            st.write(f"File 1: {bucket_path1}")
            for line in file_contents1:
                st.write(decode_sequence(line))
        with col2:
            st.write(f"File 2: {bucket_path2}")
            for line in file_contents2:
                st.write(decode_sequence(line))


if __name__ == "__main__":
    main()
