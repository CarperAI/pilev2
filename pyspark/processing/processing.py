import argparse

from faker import Faker
from functools import partial
from helpers import default_arguments, get_file_paths
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, MinHashLSH
from pyspark.sql.functions import lit, udf
from pyspark.sql.types import StringType, FloatType
from scrubadub import Scrubber
from scrubadub.detectors import CredentialDetector, TwitterDetector, UrlDetector
from scrubadub.filth import CreditCardFilth, EmailFilth, PhoneFilth, SocialSecurityNumberFilth
from spark_session_builder import build_spark_session
from squeakily.clean import fix_utf8_encoding
from squeakily.filter import (
    check_char_repetition,
    check_flagged_words,
    check_stop_word_ratio,
    check_compression_ratio,
    check_word_number,
)

# Setup the UDFs
def convert_to_udf(func):
    func_p = partial(func, dry_run=True)
    func_p.name = func.__name__
    return udf(func_p, FloatType())

udf_check_char_repetition = convert_to_udf(check_char_repetition)
udf_check_flagged_words = convert_to_udf(check_flagged_words)
udf_check_stop_word_ratio = convert_to_udf(check_stop_word_ratio)
udf_check_compression_ratio = convert_to_udf(check_compression_ratio)
udf_check_word_number = convert_to_udf(check_word_number)

scrubber = Scrubber()
scrubber.remove_detector(CredentialDetector)
scrubber.remove_detector(TwitterDetector)
scrubber.remove_detector(UrlDetector)

faker = Faker()
pii_cleaning = lambda x: (
    scrubber.clean(x)
            .replace("{{EMAIL}}", EmailFilth.generate(faker))
            .replace("{{PHONE}}", PhoneFilth.generate(faker))
            .replace("{{SOCIAL_SECURITY_NUMBER}}", SocialSecurityNumberFilth.generate(faker))
            .replace("{{CREDIT_CARD}}", CreditCardFilth.generate(faker))
)
udf_pii_cleaning = udf(pii_cleaning, StringType())
udf_fix_utf8_encoding = udf(fix_utf8_encoding, StringType())

parser = default_arguments()
parser.add_argument(
    "--quantile",
    type=float,
    default=0.01,
    help="The quantile to use for filtering.",
)
args = parser.parse_args()

spark = build_spark_session(f"spark://{args.main_addr}:7077", args.num_cpus_per_node, args.mem_per_node)
paths = get_file_paths(args.bucket_name, args.data_dir, args.extension)

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

def filters(df, func, name):
    print(f"Filtering {name}...")
    df = df.withColumn(name, func("text"))
    quantile = df.approxQuantile(name, [args.quantile], 0.0)[0]
    df = df.filter(df[name] > quantile)
    return df


# Clean the data
df = df.withColumn("text", udf_fix_utf8_encoding("text"))
df = df.withColumn("text", udf_pii_cleaning("text"))

# Filter the data
df = filters(df, udf_check_char_repetition, "char_repetition")
df = filters(df, udf_check_flagged_words, "flagged_words")
df = filters(df, udf_check_stop_word_ratio, "stop_word_ratio")
df = filters(df, udf_check_word_number, "word_number")
df = filters(df, udf_check_compression_ratio, "compression_ratio")

# Deduplicate the data

# Exact duplicates
pipeline = Pipeline(stages=[
    RegexTokenizer(inputCol="text", outputCol="words", pattern="[^A-Za-z_0-9]"),
    NGram(n=2, inputCol="words", outputCol="ngrams"),
)
pipeline_model = pipeline.fit(df)
df = pipeline_model.transform(df)
df = df.filter(size(col("ngrams")) > 0)

hasher = HashingTF(inputCol="ngrams", outputCol="hashes")
df = hasher.transform(df)
df = df.dropDuplicates(["hashes"])

# Similar duplicates
lsh = MinHashLSH(inputCol="hashes", outputCol="hashes_lsh", seed=42)
model = lsh.fit(df)
duplicates = model.approxSimilarityJoin(df, df, 0.5, distCol="distance").filter("datasetA.id < datasetB.id").filter("datasetA.id != datasetB.id")
df = df.filter(~df.id.isin(duplicates.select("datasetA.id").collect()))


# Write the data to disk per data source
for data_source in paths.keys():
    df.filter(df.data_source == data_source).write.parquet(f"s3a://{args.bucket_name}/{args.data_dir}/{data_source}/cleaned.parquet")
