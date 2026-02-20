from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA


def run_modeling(input_path: str = "data/normalized/"):
    spark = SparkSession.builder \
        .appName("TopicModeling") \
        .getOrCreate()

    # Load normalized data (expects a 'tokens' column)
    df = spark.read.parquet(input_path)

    # Convert tokens to feature vectors
    vectorizer = CountVectorizer(
        inputCol="tokens",
        outputCol="features",
        vocabSize=10000,
        minDF=5
    )
    vector_model = vectorizer.fit(df)
    vectorized_df = vector_model.transform(df)

    # Train LDA model
    lda = LDA(k=10, maxIter=10, featuresCol="features")
    lda_model = lda.fit(vectorized_df)

    # Show topics
    print("===== Extracted Topics =====")
    lda_model.describeTopics().show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    run_modeling()
