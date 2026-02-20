"""
Standalone LDA topic modeling using scikit-learn.
Can be run independently for batch processing.
"""
import os
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_TOPICS = int(os.environ.get("NUM_TOPICS", "10"))
TOP_WORDS = int(os.environ.get("TOP_WORDS_PER_TOPIC", "10"))


def run_modeling(texts: list[str], num_topics: int = None) -> dict:
    """
    Run LDA topic modeling on a list of texts.

    Returns dict with topics_summary and per-document results.
    """
    num_topics = num_topics or NUM_TOPICS

    if len(texts) < 5:
        return {"error": "Need at least 5 documents", "results": []}

    # Vectorize
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=5000,
        stop_words='english'
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=10
    )
    lda.fit(doc_term_matrix)

    # Get topic summaries
    topics_summary = []
    for i, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-TOP_WORDS:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        topics_summary.append({"topic_id": i, "top_words": top_words})
        logger.info(f"Topic {i}: {', '.join(top_words[:5])}")

    # Get per-document results
    topic_distributions = lda.transform(doc_term_matrix)
    results = []
    for i, dist in enumerate(topic_distributions):
        dominant = int(np.argmax(dist))
        results.append({
            "id": i,
            "dominant_topic": dominant,
            "topic_distribution": [round(float(p), 4) for p in dist],
            "top_words": topics_summary[dominant]["top_words"]
        })

    return {
        "num_topics": num_topics,
        "topics_summary": topics_summary,
        "results": results
    }


if __name__ == "__main__":
    # Example
    sample = [
        "Machine learning transforms artificial intelligence",
        "Stock market shows gains in technology sector",
        "Climate change impacts are becoming severe",
        "New programming languages for data science",
        "Healthcare innovations improve patient outcomes",
    ] * 10

    result = run_modeling(sample, num_topics=3)
    print(f"Topics: {len(result['topics_summary'])}")
    for t in result['topics_summary']:
        print(f"  Topic {t['topic_id']}: {', '.join(t['top_words'][:5])}")
