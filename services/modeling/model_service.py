"""
Modeling service - LDA topic modeling with drift detection using scikit-learn.
"""
import os
import sys
sys.path.insert(0, "/app")

import logging
import numpy as np
from collections import deque
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import jensenshannon

from shared.config import KAFKA_BROKER, KAFKA_TOPICS, BATCH_SIZE as DEFAULT_BATCH_SIZE
from shared.kafka_utils import create_consumer, create_producer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
BATCH_SIZE = int(os.environ.get("MODEL_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
NUM_TOPICS = int(os.environ.get("NUM_TOPICS", "10"))
TOP_WORDS = int(os.environ.get("TOP_WORDS_PER_TOPIC", "10"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "sklearn-lda-v1")
DRIFT_WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "200"))
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.15"))


class TopicModeler:
    """LDA topic modeling using scikit-learn."""

    def __init__(self, num_topics: int, top_words: int):
        self.num_topics = num_topics
        self.top_words = top_words
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=5000,
            stop_words='english'
        )
        self.lda = LatentDirichletAllocation(
            n_components=num_topics,
            learning_method='online',
            random_state=42,
            max_iter=10
        )
        self.is_fitted = False
        self.feature_names = []

    def fit_batch(self, texts: list[str]) -> bool:
        """Fit or partial_fit on a batch of texts."""
        if len(texts) < 5:
            logger.warning(f"Batch too small ({len(texts)}), skipping")
            return False

        try:
            if not self.is_fitted:
                doc_term_matrix = self.vectorizer.fit_transform(texts)
                self.feature_names = self.vectorizer.get_feature_names_out()
                self.lda.fit(doc_term_matrix)
                self.is_fitted = True
                logger.info(f"Model fitted on {len(texts)} docs, vocab: {len(self.feature_names)}")
            else:
                doc_term_matrix = self.vectorizer.transform(texts)
                self.lda.partial_fit(doc_term_matrix)
                logger.info(f"Model updated with {len(texts)} docs")
            return True
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            return False

    def get_topics(self, text: str) -> dict:
        """Get topic distribution for a single text."""
        if not self.is_fitted:
            return {"dominant_topic": -1, "topic_distribution": [], "top_words": []}

        try:
            doc_term_matrix = self.vectorizer.transform([text])
            topic_dist = self.lda.transform(doc_term_matrix)[0]
            dominant_topic = int(np.argmax(topic_dist))
            top_words = self._get_top_words(dominant_topic)

            return {
                "dominant_topic": dominant_topic,
                "topic_distribution": [round(float(p), 4) for p in topic_dist],
                "top_words": top_words
            }
        except Exception as e:
            logger.error(f"Error getting topics: {e}")
            return {"dominant_topic": -1, "topic_distribution": [], "top_words": []}

    def _get_top_words(self, topic_idx: int) -> list[str]:
        """Get top words for a topic."""
        if not self.is_fitted or topic_idx < 0:
            return []
        topic = self.lda.components_[topic_idx]
        top_indices = topic.argsort()[-self.top_words:][::-1]
        return [self.feature_names[i] for i in top_indices]

    def get_all_topics_summary(self) -> list[dict]:
        """Get summary of all topics."""
        if not self.is_fitted:
            return []
        return [
            {"topic_id": i, "top_words": self._get_top_words(i)}
            for i in range(self.num_topics)
        ]


class DriftDetector:
    """Detects topic drift using Jensen-Shannon divergence."""

    def __init__(self, num_topics: int, window_size: int, threshold: float):
        self.num_topics = num_topics
        self.window_size = window_size
        self.threshold = threshold
        self.current_window = []
        self.previous_distribution = None
        self.window_count = 0
        self.drift_events = []

    def add_topic(self, dominant_topic: int) -> dict:
        """Add topic and check for drift."""
        if dominant_topic < 0:
            return {"drift_detected": False, "window_id": self.window_count,
                    "js_divergence": 0.0, "drift_magnitude": "none"}

        self.current_window.append(dominant_topic)

        result = {
            "drift_detected": False,
            "window_id": self.window_count,
            "js_divergence": 0.0,
            "drift_magnitude": "none"
        }

        if len(self.current_window) >= self.window_size:
            drift_info = self._check_drift()
            result.update(drift_info)
            self.current_window = []
            self.window_count += 1

        return result

    def _check_drift(self) -> dict:
        """Check drift between windows."""
        counts = np.zeros(self.num_topics)
        for t in self.current_window:
            if 0 <= t < self.num_topics:
                counts[t] += 1
        current_dist = (counts + 1e-10) / (len(self.current_window) + self.num_topics * 1e-10)

        if self.previous_distribution is None:
            self.previous_distribution = current_dist
            return {"drift_detected": False, "js_divergence": 0.0, "drift_magnitude": "none"}

        js_div = jensenshannon(self.previous_distribution, current_dist)
        drift_detected = js_div > self.threshold

        if js_div < 0.05:
            magnitude = "none"
        elif js_div < 0.10:
            magnitude = "low"
        elif js_div < 0.20:
            magnitude = "medium"
        else:
            magnitude = "high"

        if drift_detected:
            self.drift_events.append({
                "window_id": self.window_count,
                "timestamp": datetime.utcnow().isoformat(),
                "js_divergence": round(float(js_div), 4),
                "magnitude": magnitude
            })
            logger.warning(f"DRIFT DETECTED window {self.window_count}: JS={js_div:.4f} ({magnitude})")

        self.previous_distribution = current_dist
        return {
            "drift_detected": drift_detected,
            "js_divergence": round(float(js_div), 4),
            "drift_magnitude": magnitude
        }

    def get_summary(self) -> dict:
        return {
            "total_windows": self.window_count,
            "drift_events_count": len(self.drift_events),
            "drift_events": self.drift_events[-10:]
        }


def main():
    logger.info("Starting modeling service (sklearn LDA + Drift Detection)...")
    logger.info(f"Config: {NUM_TOPICS} topics, batch {BATCH_SIZE}, drift window {DRIFT_WINDOW_SIZE}")

    consumer = create_consumer(KAFKA_TOPICS["normalized"], "modeling-group")
    producer = create_producer()
    modeler = TopicModeler(NUM_TOPICS, TOP_WORDS)
    drift_detector = DriftDetector(NUM_TOPICS, DRIFT_WINDOW_SIZE, DRIFT_THRESHOLD)

    message_buffer = deque()
    processed_count = 0

    logger.info(f"Consuming from {KAFKA_TOPICS['normalized']}, producing to {KAFKA_TOPICS['modeled']}")

    try:
        for message in consumer:
            try:
                msg = message.value
                message_buffer.append(msg)

                if len(message_buffer) >= BATCH_SIZE:
                    # Fit model on batch
                    texts = [m.get("text", "") for m in message_buffer]
                    modeler.fit_batch(texts)

                    # Process each message
                    while message_buffer:
                        buffered_msg = message_buffer.popleft()
                        topics = modeler.get_topics(buffered_msg.get("text", ""))
                        drift = drift_detector.add_topic(topics["dominant_topic"])

                        output = {
                            "id": buffered_msg.get("id"),
                            "title": buffered_msg.get("title", ""),
                            "text": buffered_msg.get("text", ""),
                            "score": float(buffered_msg.get("score", 0.0)),
                            "original_length": int(buffered_msg.get("original_length", 0) or 0),
                            "normalized_length": int(buffered_msg.get("normalized_length", 0) or 0),
                            "model_version": MODEL_VERSION,
                            "dominant_topic": int(topics["dominant_topic"]),
                            "topic_distribution": topics["topic_distribution"],
                            "top_words": list(topics["top_words"]),
                            "window_id": int(drift["window_id"]),
                            "drift_detected": bool(drift["drift_detected"]),
                            "js_divergence": float(drift["js_divergence"]),
                            "drift_magnitude": str(drift["drift_magnitude"])
                        }

                        producer.send(KAFKA_TOPICS["modeled"], value=output)
                        processed_count += 1

                    producer.flush()
                    logger.info(f"Processed {processed_count} messages")

                    # Log topics periodically
                    if processed_count % 500 == 0:
                        for t in modeler.get_all_topics_summary()[:3]:
                            logger.info(f"Topic {t['topic_id']}: {', '.join(t['top_words'][:5])}")

            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                continue

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        # Process remaining
        if message_buffer:
            texts = [m.get("text", "") for m in message_buffer]
            if len(texts) >= 5:
                modeler.fit_batch(texts)

            while message_buffer:
                buffered_msg = message_buffer.popleft()
                topics = modeler.get_topics(buffered_msg.get("text", ""))
                drift = drift_detector.add_topic(topics["dominant_topic"])

                output = {
                    "id": buffered_msg.get("id"),
                    "title": buffered_msg.get("title", ""),
                    "text": buffered_msg.get("text", ""),
                    "score": float(buffered_msg.get("score", 0.0)),
                    "original_length": int(buffered_msg.get("original_length", 0) or 0),
                    "normalized_length": int(buffered_msg.get("normalized_length", 0) or 0),
                    "model_version": MODEL_VERSION,
                    "dominant_topic": int(topics["dominant_topic"]),
                    "topic_distribution": topics["topic_distribution"],
                    "top_words": list(topics["top_words"]),
                    "window_id": int(drift["window_id"]),
                    "drift_detected": bool(drift["drift_detected"]),
                    "js_divergence": float(drift["js_divergence"]),
                    "drift_magnitude": str(drift["drift_magnitude"])
                }

                producer.send(KAFKA_TOPICS["modeled"], value=output)
                processed_count += 1

        summary = drift_detector.get_summary()
        logger.info(f"Final: {summary['drift_events_count']} drift events in {summary['total_windows']} windows")

        producer.flush()
        producer.close()
        consumer.close()
        logger.info(f"Shut down. Total: {processed_count}")


if __name__ == "__main__":
    main()
