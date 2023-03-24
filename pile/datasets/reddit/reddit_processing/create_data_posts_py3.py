"""A Dataflow script for creating datasets from reddit.

For usage see README.md.
"""
# Adapted from the awesome folk at PolyAI: https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit

# download the reddit data from https://files.pushshift.io/reddit/submissions/ and upload to GCS then run this script
# 2005-2022 running command:
# python create_data_posts_py3.py --runner DataflowRunner --input_type gcs --input_gcs_dir GCS_DIR --temp_location GCS_DIR --staging_location GCS_DIR --project august-clover-363917 --output_dir GCS_DIR --coding_reddits_output_dir GCS_DIR  --setup_file ./setup.py --dataset_format LM_DATAFORMAT

                                                                                                  
import argparse
import json
import logging
import os
from lm_dataformat import *
import apache_beam as beam
from functools import partial
from apache_beam import pvalue
from apache_beam.io import Read, ReadFromBigQuery
from apache_beam.io.textio import WriteToText, ReadFromText
from apache_beam.io.tfrecordio import WriteToTFRecord
from apache_beam.io.parquetio import WriteToParquet
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.io import iobase
from apache_beam.transforms import ptransform
from apache_beam.io.filesystem import CompressionTypes
import ujson
import regex as re

_TF_FORMAT = "TF"
_JSON_FORMAT = "JSON"
_LM_DATAFORMAT = "LM_DATAFORMAT"
_PARQUET_FORMAT = "PARQUET"

PROJECT = "INSERT YOUR PROJECT HERE"
BUCKET="INSERT YOUR BUCKET HERE"


def setup_google_cloud_credentials():
    """Sets up Google Cloud credentials."""
    # if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(
            "INSERT YOUR GOOGLE CLOUD CREDENTIALS HERE")




def _parse_args(argv=None):
    """Parse command line arguments."""

    def _positive_int(value):
        """Define a positive integer ArgumentParser type."""
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "Value must be positive, {} was passed.".format(value))
        return value

    parser = argparse.ArgumentParser()

    parser.add_argument("--banned_subreddits_file",
                        required=False,
                        help="a text file containing a list of banned subreddits.",   
                        default="banned_subreddits.txt"
                        )
    parser.add_argument( "--programming_subreddits_file",
                        required=False,
                        help="a text file containing a list of computer science subreddits.",
                        default="programming_subreddits.txt"
                )

    parser.add_argument(
        "--reddit_table", #reddit_comments_upto_2019  
        required=False, # reddit_sample
        default= "INSERT DATASET HERE:INSERT BIGQUERY TABLE HERE",
        help="The BigQuery table to read comments from, in "
             "project:table format.",
    )
    parser.add_argument(
        "--input_gcs_dir", #reddit_comments_upto_2019  
        required=False, # reddit_sample
        default= "INSERT GCS DIRECTORY HERE",
        help="Input google storage directory to read the data files from."
    )
    
    parser.add_argument(
        "--input_type",
        required=True,
        choices=["bigquery", "gcs"],
        help="The type of input to read from.",
    )

    parser.add_argument(
        "--output_dir",
        required=False, # reddit_sample
        # reddit_comments_upto_2019
        default= 'INSERT HERE', 
        help="Google cloud storage output directory to write the pilev2 portion of the dataset.",
    )

    parser.add_argument(
        "--coding_reddits_output_dir",
        required=False, # reddit_sample
        # reddit_comments_upto_2019
        default= 'INSERT HERE', 
        help="Google cloud storage output directory to write the code pile portion of the dataset.",
    )

    parser.add_argument(
        "--dataset_format",
        choices={_TF_FORMAT, _JSON_FORMAT, _LM_DATAFORMAT, _PARQUET_FORMAT},
        default=f"{_LM_DATAFORMAT}",
        help="The dataset format to write. 'TF' for serialized tensorflow "
             "examples in TFRecords. 'JSON' for text files with one JSON "
             "object per line. 'LM_DATAFORMAT' for the lm_dataformat library."
             " 'PARQUET' for parquet files.",
    )
    
    parser.add_argument(
        "--max_words",
        type=_positive_int,
        default=100_000, # was 127 then 512. 
        # this would enable a model with 2048 context window to attend to all 12 parents within that context window
        help="Maximum words in posts to include.",
    )
    parser.add_argument(
        "--min_words",
        type=_positive_int,
        default=15,
        help="Minimum words in posts to include.",
    )
    parser.add_argument(
        "--num_shards",
        default=500, # 1000
        type=_positive_int,
        help="The number of shards in the dataset.",
    )
    parser.add_argument(
        "--min_score",
        default=2,
        type=_positive_int,
        help="The minimum score of a post to include.",
    )
    return parser.parse_known_args(argv)


def normalize_post(post, max_words):
    def safe_str(obj):
        try: 
            return str(obj)
        except UnicodeEncodeError:
            return "[UNICODE ENCODE ERROR]"


    def _normalize_string(text):
        """Normalizes a string by removing non-printable characters and stripping
        whitespace."""
        # check if text is not None
        if text is None:
            return ""

        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.strip()
        return safe_str(text)

    def _get_num_words(text):
        """Returns the number of words in a string."""
        # check if text is not None
        if text is None:
            return 0
            
        return len(re.findall(r'\w+', text.lower()))

    def get_words(text):
        return re.findall(r'\w+', text)

    def _normalize_id(raw_id):
        import re
        """Reddit IDs start with t1_, t2_, etc. which need to be stripped."""
        # convert to string
        raw_id = str(raw_id)
        return re.sub("^t[0-9]_", "", raw_id)
    def trim(text, max_words):
        """Trims text to be at most `max_words`, without splitting apart words."""
        if _get_num_words(text) <= max_words:
            return text
        words = get_words(text)

        trimmed_text = words[:max_words + 1]
        text = " ".join(trimmed_text)

        # Trim until the last two characters are the boundary between an
        # alphanumeric character, and a non-alphanumeric character.
        while len(text) > 1 and (text[-1].isalnum() == text[-2].isalnum()):
            text = text[:-1]

        return text[:-1]
    
    from collections import namedtuple

    """Create a _Post object from a row in the BigQuery table or jsonl."""
    # represent a reddit post
    Post = namedtuple(
        "Post",
        [
            "id",
            "title",
            "author",
            "subreddit",
            "subreddit_id",
            "body",
            "body_is_trimmed",
            "created_utc",
            "score",
            # nsfw
            "over_18",
            "num_comments",

        ]
    )
    body_key = "body" if "body" in post else "selftext"
    post = Post(
        id=post['id'],
        title=_normalize_string(post['title']) if 'title' in post else None,
        author=_normalize_string(post['author']) if 'author' in post else None,
        subreddit=_normalize_string(post['subreddit']) if 'subreddit' in post else None,
        subreddit_id=_normalize_id(post['subreddit_id']) if 'subreddit_id' in post else None,
        body=trim(_normalize_string(post[body_key]), max_words),
        body_is_trimmed=_get_num_words(post[body_key]) > max_words,
        created_utc=post['created_utc'] if 'created_utc' in post else None,
        score=post['score'] if 'score' in post else None,
        over_18=post['over_18'] if 'over_18' in post else None,
        num_comments=post['num_comments'] if 'num_comments' in post else None,
    )
    # transform the post from a namedtuple into a dictionary
    post = post._asdict()
    return post

def create_examples(post, min_words, format, min_score):
    def _get_num_words(text):
        """Returns the number of words in a string."""
        # check if text is not None
        if text is None:
            return 0
        return len(re.findall(r'\w+', text.lower()))

    def _should_skip(post, min_words, min_score):

        # check if any of the fields are None
        for key in post:
            if post[key] is None:
                return True
        """
    example ["id"] = post['id']
    example ["title"] = post['title']
    example ["author"] = post['author']
    example ["subreddit"] = post['subreddit']
    example ["subreddit_id"] = post['subreddit_id']
    example ["body"] = post['body']
    example ["body_is_trimmed"] = post['body_is_trimmed']
    example ["score"] = post['score']
    example ["over_18"] = post['over_18']
    example ["num_comments"] = post['num_comments']
        """
        # return True if any of the important keys are empty
        
        if post['body_is_trimmed']:
            return True
        if post['body'] in {"[deleted]", "[removed]", "[UNICODE ENCODE ERROR]"} or \
                post['author'] in {"[deleted]", "[removed]", "[UNICODE ENCODE ERROR]"}:
            return True
        if post['subreddit'] in {"[deleted]", "[removed]", "[UNICODE ENCODE ERROR]"}:
            return True
        if _get_num_words(post['body']) < min_words:
            return True
        
        if post['over_18']:
            return True
        
        if post['score'] < min_score:
            return True

        return False

    import datetime
    """Creates serialized tensorflow examples from a reddit thread."""

    if (_should_skip(post, min_words, min_score)):
        return None

    example = {}

    example ["id"] = post['id']
    example ["title"] = post['title']
    example ["author"] = post['author']
    example ["subreddit"] = post['subreddit']
    example ["subreddit_id"] = post['subreddit_id']
    example ["body"] = post['body']
    example ["body_is_trimmed"] = post['body_is_trimmed']
    example ["score"] = post['score']
    example ["over_18"] = post['over_18']
    example ["num_comments"] = post['num_comments']
    # handle the date if string or NONE
    if post['created_utc'] is None:
        example ["created_utc"] = None
        year = None
    else:
        try:
                example ["created_utc"] = int(post['created_utc'])
                year = datetime.datetime.fromtimestamp(int(post['created_utc'])).strftime('%Y')
        except ValueError:
                example ["created_utc"] = None
                year = None
      
    example ["created_utc"] = post['created_utc']

    formatted_post_str = ''
    if example["title"] is not None:
        formatted_post_str += f"Title: {example['title']}\n"
    if example["subreddit"] is not None:
        if year is not None:
            formatted_post_str +='The text below was posted in an online community called {} in the year {}:\n'.format(
                         post['subreddit'],
                        year,
                    
                        )
        else:
            formatted_post_str +='The text below was posted in an online community called {}:\n'.format(
                        post['subreddit'], 
                        )
    formatted_post_str += f"\n"
    formatted_post_str += example['body']
    

    example['formatted_post'] = formatted_post_str

    yield example



def _features_to_serialized_tf_example(features):
    import tensorflow as tf
    """Convert a string dict to a serialized TF example.

    The dictionary maps feature names (strings) to feature values (strings).
    """
    example = tf.train.Example()
    for feature_name, feature_value in list(features.items()):
        example.features.feature[feature_name].bytes_list.value.append(
            feature_value.encode("utf-8"))
    return example.SerializeToString()



def process_records_to_lm_data(example):
    """Writes examples to a file in the language modeling data format.
    input: example is a dictionary containing the following keys:

    'id': the id of the post
    'title': the title of the post
    'author': the author of the post
    'subreddit': the subreddit of the post
    'body': the body of the post
    'body_is_trimmed': whether the body of the post was trimmed
    'created_utc': the time the post was created
    'score': the score of the post
    'formatted_post': the formatted post

    output: a dictionary containing the following keys:
    'text': the text of the post
    'meta': a dictionary containing metadata about the post
    """
    # for example in example:
    formatted_data = {}
    formatted_data['text'] = example['formatted_post']
    formatted_data['meta'] = {}
    formatted_data['meta']['source'] = 'reddit_posts'
    for k in list(example.keys()):
        if k not in ['formatted_post']:
                formatted_data['meta'][k] = example[k]
        
    formatted_data = json.dumps(formatted_data)
    return formatted_data


def _shuffle(pcollection):
    import uuid
    import apache_beam as beam
    """Shuffles the input pcollection."""
    pcollection |= "add random key" >> beam.Map(
        lambda value: (uuid.uuid4(), value))
    pcollection |= "group by key" >> beam.GroupByKey()
    pcollection |= "get shuffled values" >> beam.FlatMap(lambda t: t[1])
    return pcollection


def safe_load_json(line):
    """Loads a line of JSON, ignoring any errors."""
    try:
        return json.loads(line)
    except ValueError:
        return None

def run(argv=None, posts=None):
    """Run the beam pipeline.

    Args:
        argv: (optional) the command line flags to parse.
        comments_collection: (optional) a list of comment JSON objects to
            process. Used in unit-tests to avoid requiring a BigQuery source.
    """

    args, pipeline_args = _parse_args(argv)
    banned_subreddits_file = args.banned_subreddits_file
    # open the banned subreddits file and readlines and strip the newlines
    banned_subreddits = [line.lower().strip() for line in open(banned_subreddits_file, 'r').readlines()]
    banned_subreddits = list(set(banned_subreddits))
    programming_subreddits_file = args.programming_subreddits_file
    # open the programming subreddits file and readlines and strip the newlines
    programming_subreddits = [line.lower().strip() for line in open(programming_subreddits_file, 'r').readlines()]
    programming_subreddits = list(set(programming_subreddits))

    pipeline_options = PipelineOptions(pipeline_args, save_main_session=True)
    # pipeline_options.view_as(SetupOptions).save_main_session = True
    p = beam.Pipeline(options=pipeline_options)

    if posts is not None:
        posts = p | ("Read in-memory posts") >> beam.Create(posts)
    elif args.input_type == "bigquery":
        posts = p | ("Read " + args.reddit_table) >> Read(
            ReadFromBigQuery(table=args.reddit_table)) # was BigQuerySource
    else:
         posts = p | ("Reading " + args.input_gcs_dir) >> Read(
            ReadFromText(args.input_gcs_dir)
           )

         posts = posts | (
            "Parse JSON" >> beam.Map(safe_load_json)
           )
        #  Filtering posts that are none
         posts = posts | (
            "Filter none posts" >> beam.Filter(lambda post: post is not None)
                                    
        )

    posts |= (
        "normalize posts" >> beam.Map(
            partial(normalize_post, max_words=args.max_words)))

     #  Filtering posts that are none
    posts = posts | (
            "Filtering posts that are skipped" >> beam.Filter(lambda post: post is not None)
                                    
        )

    # thread_id_to_posts = posts | (
    #     "Key by thread id" >> beam.Map(
    #         lambda comment: (comment['thread_id'], comment)))
    # threads = thread_id_to_posts | (
    #     "Group posts by thread ID" >> beam.GroupByKey())
    # threads = threads | ("Get threads" >> beam.Map(lambda t: t[1]))

    examples = posts | (
        "Create {} examples".format(args.dataset_format) >> beam.FlatMap(
            partial(create_examples,
                    min_words=args.min_words,
                    format=args.dataset_format,
                    min_score = args.min_score,
                    )))

    examples = _shuffle(examples)

    #  Filtering examples from banned subreddits
    examples = examples | (
        "Filter examples from banned subreddits" >> beam.Filter(lambda example: 
                                example['subreddit'].lower() not in banned_subreddits)
    )

    #  Filtering examples from programming subreddits
    programming_examples = examples | (
                                "Filter examples from programming subreddits" >> beam.Filter(lambda example:
                                example['subreddit'].lower() in programming_subreddits)
    )

    #  Filtering examples from non-programming subreddits
    non_programming_examples = examples | (
                                "Filter examples from non-programming subreddits" >> beam.Filter(lambda example:    
                                example['subreddit'].lower() not in programming_subreddits)
    )


    if args.dataset_format == _JSON_FORMAT:
        file_name_suffix = ".jsonl"
        serialize_fn = json.dumps
        write_sink = WriteToText
        compression_type = CompressionTypes.AUTO
    elif args.dataset_format == _LM_DATAFORMAT:
        file_name_suffix = ".jsonl.zst"
        serialize_fn = process_records_to_lm_data
        write_sink = WriteToText
        compression_type = CompressionTypes.ZSTD
    elif args.dataset_format == _TF_FORMAT:
        assert args.dataset_format == _TF_FORMAT
        write_sink = WriteToTFRecord
        file_name_suffix = ".tfrecord"
        serialize_fn = _features_to_serialized_tf_example
        compression_type = None
    elif args.dataset_format == _PARQUET_FORMAT:
        # https://stackoverflow.com/questions/48915883/how-to-write-string-to-parquet-file-using-beam
        write_sink = WriteToParquet
        file_name_suffix = ".parquet"
        serialize_fn = json.dumps
        compression_type = None

    if compression_type is not None:

        for name, processed_examples in [("programming", programming_examples),
                ("non-programming", non_programming_examples)]:
            # if programming then write to args.coding_reddits_output_dir else write to args.output_dir
            if name == "programming":
                output_dir = args.coding_reddits_output_dir
                # get 20% of num_shards
                num_shards = int(args.num_shards * 0.2)
            else:
                output_dir = args.output_dir
                num_shards = args.num_shards
            serialized_examples = processed_examples | (
                "serialize {} examples".format(name) >> beam.Map(serialize_fn))
            (   
                
                serialized_examples | ("write " + name)
                >> write_sink (
                    "{}/{}".format(output_dir, name),
                    num_shards=args.num_shards,
                    
                    file_name_suffix=file_name_suffix,
                    
                    compression_type=compression_type,

                )
            )
    else:
        for name, processed_examples in [("programming", programming_examples),
                ("non-programming", non_programming_examples)]:
            
            # if programming then write to args.coding_reddits_output_dir else write to args.output_dir
            if name == "programming":
                output_dir = args.coding_reddits_output_dir
                # get 20% of num_shards
                num_shards = int(args.num_shards * 0.2)
            else:
                output_dir = args.output_dir
                num_shards = args.num_shards

            serialized_examples = processed_examples | (
                "serialize {} examples".format(name) >> beam.Map(serialize_fn))
            (
                serialized_examples | ("write " + name)
                >> write_sink (
                    "{}/{}".format(output_dir, name),
                    num_shards=num_shards,
                    
                    file_name_suffix=file_name_suffix,
                    

                )
            )


    result = p.run()
    result.wait_until_finish()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    setup_google_cloud_credentials() 
    run()
