import os
import sys
import findspark
findspark.init()

# Print environment variables for debugging
print("HADOOP_HOME:", os.environ.get('HADOOP_HOME'))
print("SPARK_HOME:", os.environ.get('SPARK_HOME'))
print("JAVA_HOME:", os.environ.get('JAVA_HOME'))
print("Python version:", sys.version)

from pyspark.sql import SparkSession

# Create Spark session with specific configurations
spark = SparkSession.builder \
    .appName("TestSpark") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.python.worker.memory", "1g") \
    .config("spark.driver.bindAddress", "localhost") \
    .config("spark.driver.host", "localhost") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.default.parallelism", "10") \
    .config("spark.python.worker.reuse", "true") \
    .getOrCreate()
# Test with a simple dataframe
test_data = [
    (1, "Hello"),
    (2, "World")
]
df = spark.createDataFrame(test_data, ["id", "text"])

# Show dataframe
df.show()

# Stop Spark session
spark.stop()