import time

from pathlib import Path
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, MinHashLSH
from pyspark.sql.functions import col, size
from spark_session_builder import build_spark_session

spark = build_spark_session("spark://cpu32-dy-c6i-8xlarge-1:7077", 32, 48)
data_dir = Path("/fsx/shared/pilev2/group1_filtered/UbuntuIRC")
parquets = [str(file) for file in data_dir.rglob("*.parquet")]
df = spark.read.parquet(*parquets)
df = df.repartition(1000)
start = time.time()

model1 = Pipeline(stages=[
    RegexTokenizer( # Stage 5
        pattern="[^A-Za-z_0-9]", inputCol="text", outputCol="tokens", minTokenLength=1
    ),
    NGram(n=13, inputCol="tokens", outputCol="ngrams"),
]).fit(df)

db_ngrams = model1.transform(df)
# Filter out ngrams that are size 0
db_ngrams = db_ngrams.filter(size(col("ngrams")) > 0)

# Continue with MinHashLSH
model2 = Pipeline(stages=[
    HashingTF(inputCol="ngrams", outputCol="vectors"), # Stage 7
    MinHashLSH(inputCol="vectors", outputCol="lsh", numHashTables=13) # Stage 8
]).fit(db_ngrams)

db_hashed = model2.transform(db_ngrams)


duplicates = model2.stages[-1].approxSimilarityJoin(
    db_hashed,
    db_hashed,
    0.15,
    distCol="JaccardDistance"
).filter("datasetA.id < datasetB.id").filter("datasetA.id != datasetB.id") # Stage 6

BUCKET="s-eai-neox"
PREFIX="data/codepile/duplicates"
# write duplicates to s3 without compression and make the max size 256MB per file, also add the schema to the parquet file
max_records_per_file = 256 * 1024 * 1024
(duplicates.write
    .option("maxRecordsPerFile", max_records_per_file)
    .option("compression", "none")
    .option("schema", duplicates.schema)
    .parquet(f"s3a://{BUCKET}/{PREFIX}/duplicates", mode="overwrite")) # Stage 7
print(f"Number of duplicates: {duplicates.count():,}")