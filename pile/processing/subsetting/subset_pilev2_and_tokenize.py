from datasets import load_from_disk, load_dataset
import logging
import tiktoken
import random
from tqdm import tqdm
import os
import argparse
import json
# import lm_dataformat as lmd
import pathlib

#GLOBALS
logging.basicConfig(
    level = logging.INFO
)
logger = logging.getLogger(__name__)
tokenizer_name = "cl100k_base"


def cust_log(text):
    print(text)
    logger.info(text)


def subset_dataset_fn(dataset,percentage:float=25.0):
    # if len(dataset) > int(percentage):
    cust_log(f"Original Length : {len(dataset)}")
    percentage_total = int(len(dataset)*percentage/100)
    idxs = random.sample(range(len(dataset)),percentage_total)
    total_len = len(dataset)
    dataset = dataset.select(idxs)
    cust_log(f"Slicing {percentage}% which is {percentage_total} |  {len(dataset)} out of {total_len}")
    #dataset.add_column("subset_idx",idxs) This takes painfully too long So making a quick replacement.
    cust_log(dataset)
    return dataset,idxs

def tokenize_and_count(datapoint:dict):
    """
    Tokenize and add the number of tokens of the tokenized
    """
    text = datapoint["text"]
    datapoint["input_ids"] = tokenizer.encode(text,disallowed_special=())
    datapoint["num_tokens"] = len(datapoint["input_ids"])
    return datapoint

# def write_dataset_to_lmd(dataset,output_path):
#     """
#     Write the dataset to lmd format
#     """
#     ar = lmd.Archive(str(output_path))
#     for i in tqdm(range(len(dataset)),leave=False,desc="Writing to lmd"):
#         ar.add_data(dataset[i])

#     ar.commit()
#     cust_log(f"Done writing to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--percentage",type=float,default=25.0)
    parser.add_argument("--tokenizer_name", type=str, default="cl100k_base")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    stats_dict = {}
    sub_dir = args.input_dir
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    tokenizer = tiktoken.get_encoding(args.tokenizer_name)
    sub_dir_contains = os.listdir(sub_dir)
    is_parquet = [i for i in sub_dir_contains if ".parquet" in i]
    is_arrow = [i for i in sub_dir_contains if ".arrow" in i]
    cust_log(f"is_parquet : {is_parquet}")
    cust_log(f"is_arrow : {is_arrow}")

    if len(is_arrow) > 0:
        dataset = load_from_disk(sub_dir)["train"]
    elif len(is_parquet) > 0:
        dataset = load_dataset("parquet",data_dir=sub_dir)["train"]
    cust_log(f"Loaded dataset from {sub_dir}")
    cust_log(dataset)
    subset_dataset,idx = subset_dataset_fn(dataset,args.percentage)
    cust_log(f"subsetted dataset : {subset_dataset}")
    with open(output_dir / "subset_idxs.json","w") as f:
        json.dump(idx,f)
    cust_log("Tokenizing")
    # subset_dataset = subset_dataset.select(range(100)).map(tokenize_and_count,num_proc=args.num_workers)
    # total_token = sum(subset_dataset["num_tokens"])
    # stats_dict = {"token_count" : total_token,"num_examples" : len(subset_dataset),"percentage" : args.percentage,"tokenizer" : args.tokenizer_name,"original_input__dir" : sub_dir,"subset_dir" : args.output_dir}
    # cust_log(
    #     json.dumps(stats_dict,indent=4)
    # )
    # with open(output_dir / "stats.json","w") as f:
    #     json.dump(stats_dict,f,indent=2)
    # with open(output_dir / "subset_idxs.json","w") as f:
    #     json.dump(idx,f)
    # cust_log("Saving")
    subset_dataset.to_parquet(output_dir/"subset.parquet")
    #write_dataset_to_lmd(subset_dataset,output_dir)
    cust_log("Done")