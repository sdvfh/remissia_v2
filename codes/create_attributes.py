from remissia.utils import get_path, get_spark

path = get_path()
spark = get_spark()

file_csv = list(path["datasets"].rglob("*.csv"))[0]
df = spark.read.csv(str(file_csv), sep=",", header=True)
df.createOrReplaceTempView("SEER")

query = (path["queries"] / "pre_processing_1.sql").read_text()
df = spark.sql(query)
df.createOrReplaceTempView("pre_processing_1")
print(len(df.columns))

query = (path["queries"] / "pre_processing_2.sql").read_text()
df = spark.sql(query)
print(len(df.columns))

df.coalesce(1).write.parquet(str(path["dataset"]), mode="overwrite")
