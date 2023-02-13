from file_utils import DataConverter
import argparse
import os
from pathlib import Path


# create main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--input_format", type=str, required=False, default="parquet", help="Input format of the data")
    args = parser.parse_args()

    # create data converter object
    data_converter = DataConverter(args.data_dir, args.output_dir, args.input_format)

    # convert to lm_dataformat
    data_converter.convert_to_lm_dataformat()

if __name__ == "__main__":
    main()
    