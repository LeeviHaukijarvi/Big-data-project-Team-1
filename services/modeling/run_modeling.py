"""
PySpark LDA topic modeling module.
"""
import os
import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover
from pyspark.ml.clustering import LDA
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# Configuration
NUM_TOPICS = int(os.environ.get("NUM_TOPICS", "10"))
MAX_ITER = int(os.environ.get("LDA_MAX_ITER", "10"))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "10000"))


def get_spark_session():
    """Create or get SparkSession."""
    return SparkSession.builder \
        .appName("TopicModeling") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()


def run_modeling(input_path: str, output_path: str = None):
    """
    Run LDA topic modeling on parquet data.

    Args:
        input_path: Path to input parquet with 'text' column
        output_path: Optional path to save results

    Returns:
        Pandas DataFrame with topic assignments, or None on error
    """
    spark = None
    try:
        spark = get_spark_session()
        spark.sparkContext.setLogLevel("WARN")

        logger.info(f"Loading data from {input_path}")
        df = spark.read.parquet(input_path)

        # Check if tokens column exists, if not create it
        if 'tokens' not in df.columns:
            # Tokenize text
            tokenizer = Tokenizer(inputCol="text", outputCol="raw_tokens")
            df = tokenizer.transform(df)

            # Remove stop words
            remover = StopWordsRemover(inputCol="raw_tokens", outputCol="tokens")
            df = remover.transform(df)

        # Filter empty token arrays
        df = df.filter(F.size(F.col("tokens")) > 0)

        if df.count() < 5:
            logger.warning("Not enough documents for LDA")
            return None

        # Convert tokens to feature vectors
        vectorizer = CountVectorizer(
            inputCol="tokens",
            outputCol="features",
            vocabSize=VOCAB_SIZE,
            minDF=2
        )
        vector_model = vectorizer.fit(df)
        vectorized_df = vector_model.transform(df)

        # Train LDA model
        logger.info(f"Training LDA with {NUM_TOPICS} topics...")
        lda = LDA(
            k=NUM_TOPICS,
            maxIter=MAX_ITER,
            featuresCol="features",
            seed=42
        )
        lda_model = lda.fit(vectorized_df)

        # Get topic assignments
        topics_df = lda_model.transform(vectorized_df)

        # Extract dominant topic
        @F.udf("integer")
        def get_dominant_topic(distribution):
            if distribution is None:
                return -1
            return int(distribution.argmax())

        @F.udf("array<double>")
        def distribution_to_list(distribution):
            if distribution is None:
                return []
            return distribution.toArray().tolist()

        result_df = topics_df.withColumn(
            "dominant_topic", get_dominant_topic(F.col("topicDistribution"))
        ).withColumn(
            "topic_distribution", distribution_to_list(F.col("topicDistribution"))
        )

        # Select relevant columns
        output_columns = ["id", "text", "dominant_topic", "topic_distribution", "top_words"]
        if "title" in result_df.columns:
            output_columns.insert(1, "title")
        if "score" in result_df.columns:
            output_columns.append("score")
        if "original_length" in result_df.columns:
            output_columns.append("original_length")
        if "normalized_length" in result_df.columns:
            output_columns.append("normalized_length")

        # Keep only existing columns
        output_columns = [c for c in output_columns if c in result_df.columns]
        result_df = result_df.select(output_columns)

        # Extract topic words
        logger.info("===== Extracted Topics =====")
        vocab = vector_model.vocabulary
        topics = lda_model.describeTopics(maxTermsPerTopic=10).collect()

        # Build topic_id -> top_words mapping
        topic_words_map = {}
        for topic in topics:
            topic_words = [vocab[i] for i in topic.termIndices]
            topic_words_map[topic.topic] = topic_words
            logger.info(f"Topic {topic.topic}: {', '.join(topic_words[:5])}")

        # Add top_words column based on dominant topic
        def get_topic_words(topic_id):
            return topic_words_map.get(topic_id, [])

        get_topic_words_udf = F.udf(get_topic_words, "array<string>")
        result_df = result_df.withColumn(
            "top_words", get_topic_words_udf(F.col("dominant_topic"))
        )

        # Convert to pandas and return
        pandas_df = result_df.toPandas()

        if output_path:
            result_df.write.mode("overwrite").parquet(output_path)
            logger.info(f"Results saved to {output_path}")

        return pandas_df

    except Exception as e:
        logger.error(f"Error in run_modeling: {e}", exc_info=True)
        return None

    finally:
        if spark:
            spark.stop()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        result = run_modeling(input_path, output_path)
        if result is not None:
            print(f"Processed {len(result)} documents")
    else:
        print("Usage: python run_modeling.py <input_path> [output_path]")
