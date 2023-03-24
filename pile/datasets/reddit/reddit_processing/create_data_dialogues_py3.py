"""A Dataflow script for creating datasets from reddit.

For usage see README.md.
"""
# Adapted from the awesome folk at PolyAI: https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit

# Running commands:
    # Run using bigquery to get Reddit data from 2005-2019
# python create_data_dialogues_py3.py --runner DataflowRunner --input_type bigquery --temp_location GCS_PATH --staging_location GCS_PATH --project PROJECT_NAME --dataset_format LM_DATAFORMAT

# python create_data_dialogues_py3.py --runner DataflowRunner --input_type bigquery --temp_location GCS_PATH --staging_location GCS --project PROJECT_NAME --dataset_format LM_DATAFORMAT

# Then download all data from 2020-2022 from https://files.pushshift.io/reddit/comments/ and https://files.pushshift.io/reddit/submissions/ and upload to GCS.
# Then run the following commands to create the data from 2020-2022
# 2020-2022 running command:
# python create_data_dialogues_py3.py --runner DataflowRunner --input_type gcs --input_gcs_dir GCS_PATH --temp_location GCS_PATH --staging_location GCS_PATH --project PROJECT_NAME --output_dir GCS_PATH --setup_file ./setup.py --dataset_format LM_DATAFORMAT
# Reddit stops end at 2022-10 include that for comments and posts ends at 2022-10-31

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

_TF_FORMAT = "TF"
_JSON_FORMAT = "JSON"
_LM_DATAFORMAT = "LM_DATAFORMAT"
_PARQUET_FORMAT = "PARQUET"

PROJECT = "INSERT_PROJECT_HERE"
BUCKET="INSERT_BUCKET_HERE"


# class LMDataSink(iobase.Sink):
#     """A Beam Sink for writing LMDataFormat data."""

#     def __init__(self, output_dir, num_shards):
#         super().__init__()
#         self.output_dir = output_dir
#         self.num_shards = num_shards
#         self.archive = Archive(self.output_dir)
#         self.archive = None


#     def initialize_write(self):
#         return 
    
#     def open_writer(self, uid):
#         return LMDataFormatWriter(self.archive)
    
#     def finalize_write(self):
#         self.archive.commit()
#         return

# """Writes to LMDataformat files. Code adapted from
#     The Archive class from LM_dataformat. https://github.com/leogao2/lm_dataformat/blob/master/lm_dataformat/__init__.py#L329
# """
# class LMDataFormatWriter(iobase.Writer):

#     def __init__(self, archive):
#         self.archive = archive
#         self._current_file = None

#     def write(self, record):
#         text, meta = record['text'], record['meta']
#         self.archive.add_data(text , meta)

#     def close(self):
#         self.archive.commit()
#         return


# class WriteToLMDataFormat(ptransform.PTransform):
#     def __init__(self, output_dir, num_shards):
#         super().__init__()
#         self.output_dir = output_dir
#         self.num_shards = num_shards
        

#     def expand(self, pcoll):
#         return pcoll | iobase.Write(
#                 LMDataSink(self.output_dir, self.num_shards)
#                 )
    
        


def setup_google_cloud_credentials():
    """Sets up Google Cloud credentials."""
    # if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(
            "INSERT_PATH_TO_GOOGLE_APPLICATION_CREDENTIALS_HERE")




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
        default= "INSERT_PROJECT_HERE:INSERT_DATASET_HERE.INSERT_TABLE_HERE",
        help="The BigQuery table to read comments from, in "
             "project:table format.",
    )
    parser.add_argument(
        "--input_gcs_dir", #reddit_comments_upto_2019  
        required=False, # reddit_sample
        default= "INSERT_BUCKET_HERE/INSERT_PATH_HERE",
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
        default= 'INSERT', 
        help="Google cloud storage output directory to write the dataset.",
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
        "--parent_depth",
        type=_positive_int,
        default=12,
        help="How many parent comments to consider.",
    )
    
    parser.add_argument(
        "--max_length",
        type=_positive_int,
        default=768, # was 127 then 512. 
        # this would enable a model with 2048 context window to attend to all 12 parents within that context window
        help="Maximum letters in comments to include.",
    )
    parser.add_argument(
        "--min_length",
        type=_positive_int,
        default=9,
        help="Minimum letters in comments to include.",
    )
    parser.add_argument(
        "--num_shards",
        default=750, # 1000
        type=_positive_int,
        help="The number of shards in the dataset.",
    )
    return parser.parse_known_args(argv)


def normalize_comment(comment, max_length):
    def safe_str(obj):
        try: 
            return str(obj)
        except UnicodeEncodeError:
            return "[UNICODE ENCODE ERROR]"

    def _normalize_id(raw_id):
        import re
        """Reddit IDs start with t1_, t2_, etc. which need to be stripped."""
        return re.sub("^t[0-9]_", "", raw_id)


    def _normalize_string(text):
        """Normalizes a string by removing non-printable characters and stripping
        whitespace."""
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.strip()
        return safe_str(text)


    def trim(text, max_length):
        """Trims text to be at most `max_length`, without splitting apart words."""
        if len(text) <= max_length:
            return text

        text = text[:max_length + 1]

        # Trim until the last two characters are the boundary between an
        # alphanumeric character, and a non-alphanumeric character.
        while len(text) > 1 and (text[-1].isalnum() == text[-2].isalnum()):
            text = text[:-1]

        return text[:-1]

    from collections import namedtuple
    """Create a _Comment object from a row in the BigQuery table."""
    # Represent a reddit comment.
    Comment = namedtuple(
    "Comment",
        [
            "id",
            "thread_id",
            "parent_id",
            "body",
            "body_is_trimmed",
            "author",
            "subreddit",
            "created_utc",
            "score",
            "link_id",


        ]
    )
    
    comment = Comment(
        id=comment['id'],
        thread_id=_normalize_id(comment['link_id']),
        parent_id=_normalize_id(comment['parent_id']),
        body=trim(_normalize_string(comment['body']), max_length),
        body_is_trimmed=len(comment['body']) > max_length,
        author=_normalize_string(comment['author']),
        subreddit=_normalize_string(comment['subreddit']),
        created_utc=comment['created_utc'],
        score=comment['score'],
        link_id=comment['link_id'],

    )
    # transform the comment from a namedtuple into a dictionary
    comment = comment._asdict()
    return comment

def normalize_post(post, max_length):
    def safe_str(obj):
        try: 
            return str(obj)
        except UnicodeEncodeError:
            return "[UNICODE ENCODE ERROR]"


    def _normalize_string(text):
        """Normalizes a string by removing non-printable characters and stripping
        whitespace."""
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.strip()
        return safe_str(text)


    def trim(text, max_length):
        """Trims text to be at most `max_length`, without splitting apart words."""
        if len(text) <= max_length:
            return text

        text = text[:max_length + 1]

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
            "body",
            "body_is_trimmed",
            "created_utc",
            "score",

        ]
    )
    post = Post(
        id=post['id'],
        title=_normalize_string(post['title']),
        author=_normalize_string(post['author']),
        subreddit=_normalize_string(post['subreddit']),
        body=trim(_normalize_string(post['body']), max_length),
        body_is_trimmed=len(post['body']) > max_length,
        created_utc=post['created_utc'],
        score=post['score'],

    )
    # transform the post from a namedtuple into a dictionary
    post = post._asdict()
    return post




# TODO: Add post to the start of the thread.
def create_examples(thread, parent_depth, min_length, format):
    
    def _should_skip(comment, min_length):
        # TODO: Skip non-English comments.

        if comment['body_is_trimmed']:
            return True
        if comment['body'] in {"[deleted]", "[removed]", "[UNICODE ENCODE ERROR]"} or \
                comment['author'] in {"[deleted]", "[removed]", "[UNICODE ENCODE ERROR]"}:
            return True
        if comment['subreddit'] in {"[deleted]", "[removed]", "[UNICODE ENCODE ERROR]"}:
            return True
        if len(comment['body']) < min_length:
            return True
        return False

    def linear_paths(id_to_comment, parent_depth):
        from collections import defaultdict
        """Gets all linear paths of comments and replies from the thread.

        Each linear path is guaranteed to have at least two comments in it.
        """
        paths = []
        seen_ids = set()
        id_to_children = defaultdict(list)
        for comment_id, comment in list(id_to_comment.items()):
            id_to_children[comment['parent_id']].append(comment_id)
            if comment['parent_id'] not in id_to_comment:
                paths.append([comment_id])
                seen_ids.add(comment_id)
        # paths start with root comments
        while paths:
            new_paths = []
            for path in paths:
                last_id = path[-1]
                for child_id in id_to_children[last_id]:
                    if child_id in seen_ids:
                        # Prevent infinite loops.
                        continue
                    seen_ids.add(child_id)
                    new_path = path[-parent_depth:] + [child_id]
                    new_paths.append(new_path)
                    
            
            # yield all unique paths at the tallest depth
            if len(paths) > 0:
                for path in paths:
                    root_node = path[0]
                    last_node = path[-1]
                    # check if the path was removed from the new paths
                    # if it was, then it is the longest path and needs to be yielded
                    if root_node not in [p[0] for p in new_paths]:       
                        yield path
                    elif last_node not in [p[-2] for p in new_paths]:
                        yield path
            paths = new_paths

    import datetime
    """Creates serialized tensorflow examples from a reddit thread."""
    id_to_comment = {comment['id']: comment for comment in list(thread)}

    for linear_path in [path for path in linear_paths(id_to_comment, parent_depth) if len(path) >= 2]:
        response = id_to_comment[linear_path[-1]]
        context = id_to_comment[linear_path[-2]]  # guaranteed to exist.

        if (_should_skip(response, min_length)
                or _should_skip(context, min_length)):
            continue

        example = {}
        example['subreddit'] = response['subreddit']
        example['thread_id'] = response['thread_id']
        example['context_author'] = context['author']
        example['response_author'] = response['author']
        example['context'] = context['body']
        example['context_score'] = context['score']
        example['response'] = response['body']
        example ['score'] = response['score']
        
        # adding meta data
        utterance_list = []

        for i, parent_id in enumerate(linear_path[:-2]):
            if i >= parent_depth:
                break
            try:
                example['context/{}'.format(i)] = id_to_comment[parent_id]['body']
                example['score/{}'.format(i)] = id_to_comment[parent_id]['score']
                utterance_list.append({"utterance": id_to_comment[parent_id]['body'], "score": id_to_comment[parent_id]['score'], "author": id_to_comment[parent_id]['author']})
            except IndexError:
                break
        
        utterance_list.append({"utterance": context['body'], "score": context['score'], "author": context['author']})
        utterance_list.append({"utterance": response['body'], "score": response['score'], "author": response['author']})
        example['conversation_sequence'] = utterance_list

        conv_str = ''
        conv_str +='In {}, the conversation below took place in an online community called {}:\n'.format(
                    datetime.datetime.fromtimestamp(response['created_utc']).strftime('%Y'),
                    response['subreddit'], 
                    )

        # need to reverse it to get the order of the parents
        for i, parent_id in enumerate(linear_path[:-2]):
            if i >= parent_depth:
                break
            conv_str += '{}: {}\n'.format(id_to_comment[parent_id]['author'], id_to_comment[parent_id]['body'])   

    
        conv_str += '{}: {}\n'.format(context['author'], context['body'])
        conv_str += '{}: {}'.format(response['author'], response['body'])
        conv_str += '\n\n\n'
        # encode all strings with utf-8
        conv_str = conv_str
        example['conversational_format'] = conv_str

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
    input: examples: a list of dicts, each with subreddit, thread_id, context_author
    response_author, context, response, score, and conversational_format.
    output: a list of dictionaries with keys text and meta.
    """
    # for example in example:
    formatted_data = {}
    formatted_data['text'] = example['conversational_format']
    formatted_data['meta'] = {}
    formatted_data['meta']['source'] = 'reddit'
    for k in list(example.keys()):
        if k not in ['conversational_format']:
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

def run(argv=None, comments=None):
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

    if comments is not None:
        comments = p | ("Read in-memory comments") >> beam.Create(comments)
    elif args.input_type == "bigquery":
        comments = p | ("Read " + args.reddit_table) >> Read(
            ReadFromBigQuery(table=args.reddit_table)) # was BigQuerySource
    else:
         comments = p | ("Reading " + args.input_gcs_dir) >> Read(
            ReadFromText(args.input_gcs_dir)
           )

         comments = comments | (
            "Parse JSON" >> beam.Map(safe_load_json)
           )
        #  Filtering comments that are none
         comments = comments | (
            "Filter none comments" >> beam.Filter(lambda comment: comment is not None)
                                    
        )

    comments |= (
        "normalize comments" >> beam.Map(
            partial(normalize_comment, max_length=args.max_length)))

    thread_id_to_comments = comments | (
        "Key by thread id" >> beam.Map(
            lambda comment: (comment['thread_id'], comment)))
    threads = thread_id_to_comments | (
        "Group comments by thread ID" >> beam.GroupByKey())
    threads = threads | ("Get threads" >> beam.Map(lambda t: t[1]))

    examples = threads | (
        "Create {} examples".format(args.dataset_format) >> beam.FlatMap(
            partial(create_examples,
                    parent_depth=args.parent_depth,
                    min_length=args.min_length,
                    format=args.dataset_format,
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
            
            serialized_examples = processed_examples | (
                "serialize {} examples".format(name) >> beam.Map(serialize_fn))
            (
                serialized_examples | ("write " + name)
                >> write_sink (
                    "{}/{}".format(args.output_dir, name),
                    num_shards=args.num_shards,
                    
                    file_name_suffix=file_name_suffix,
                    
                    compression_type=compression_type,

                )
            )
    else:
        for name, processed_examples in [("programming", programming_examples),
                ("non-programming", non_programming_examples)]:
            

            serialized_examples = processed_examples | (
                "serialize {} examples".format(name) >> beam.Map(serialize_fn))
            (
                serialized_examples | ("write " + name)
                >> write_sink (
                    "{}/{}".format(args.output_dir, name),
                    num_shards=args.num_shards,
                    
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
