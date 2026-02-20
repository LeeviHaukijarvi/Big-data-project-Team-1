"""
Modeling service - LDA topic modeling with drift detection.

Consumes normalized text from Kafka, applies LDA topic modeling,
detects topic drift between time windows, and produces enriched messages.
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
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from shared.config import KAFKA_BROKER, KAFKA_TOPICS, BATCH_SIZE as DEFAULT_BATCH_SIZE
from shared.kafka_utils import create_consumer, create_producer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration (from env or defaults)
NUM_TOPICS = int(os.environ.get("NUM_TOPICS", "10"))
BATCH_SIZE = int(os.environ.get("MODEL_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
TOP_WORDS_PER_TOPIC = int(os.environ.get("TOP_WORDS_PER_TOPIC", "10"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "lda-v1")

# Drift detection configuration
DRIFT_WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "200"))
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.15"))


class TopicModeler:
    """LDA-based topic modeling with batch updates."""

    def __init__(self, num_topics: int = NUM_TOPICS):
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(
            max_df=0.95,  # Ignore terms in >95% of docs
            min_df=2,     # Ignore terms in <2 docs
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

    def fit_batch(self, texts: list[str]) -> None:
        """Fit or partial_fit the model on a batch of texts."""
        if len(texts) < 5:
            logger.warning(f"Batch too small ({len(texts)}), skipping model update")
            return

        try:
            if not self.is_fitted:
                # Initial fit
                doc_term_matrix = self.vectorizer.fit_transform(texts)
                self.feature_names = self.vectorizer.get_feature_names_out()
                self.lda.fit(doc_term_matrix)
                self.is_fitted = True
                logger.info(f"Initial model fit on {len(texts)} documents")
            else:
                # Partial fit for online learning
                doc_term_matrix = self.vectorizer.transform(texts)
                self.lda.partial_fit(doc_term_matrix)
                logger.info(f"Model updated with {len(texts)} documents")
        except Exception as e:
            logger.error(f"Error fitting model: {e}")

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
        top_indices = topic.argsort()[-TOP_WORDS_PER_TOPIC:][::-1]
        return [self.feature_names[i] for i in top_indices]

    def get_all_topics_summary(self) -> list[dict]:
        """Get summary of all topics with their top words."""
        if not self.is_fitted:
            return []

        summaries = []
        for idx in range(self.num_topics):
            summaries.append({
                "topic_id": idx,
                "top_words": self._get_top_words(idx)
            })
        return summaries


class DriftDetector:
    """Detects topic drift between time windows using Jensen-Shannon divergence."""

    def __init__(self, num_topics: int, window_size: int = DRIFT_WINDOW_SIZE,
                 threshold: float = DRIFT_THRESHOLD):
        self.num_topics = num_topics
        self.window_size = window_size
        self.threshold = threshold
        self.current_window = []
        self.previous_distribution = None
        self.window_count = 0
        self.drift_events = []

    def add_topic(self, dominant_topic: int) -> dict:
        """Add a topic assignment and check for drift when window completes."""
        if dominant_topic < 0:
            return {"drift_detected": False, "window_id": self.window_count}

        self.current_window.append(dominant_topic)

        result = {
            "drift_detected": False,
            "window_id": self.window_count,
            "js_divergence": 0.0,
            "drift_magnitude": "none"
        }

        # Check for drift when window is complete
        if len(self.current_window) >= self.window_size:
            drift_info = self._check_drift()
            result.update(drift_info)
            self._start_new_window()

        return result

    def _get_distribution(self, topics: list[int]) -> np.ndarray:
        """Calculate topic distribution from list of topic assignments."""
        counts = np.zeros(self.num_topics)
        for t in topics:
            if 0 <= t < self.num_topics:
                counts[t] += 1
        # Add small epsilon to avoid division by zero
        dist = (counts + 1e-10) / (len(topics) + self.num_topics * 1e-10)
        return dist

    def _check_drift(self) -> dict:
        """Check for drift between current and previous window."""
        current_dist = self._get_distribution(self.current_window)

        if self.previous_distribution is None:
            self.previous_distribution = current_dist
            return {"drift_detected": False, "js_divergence": 0.0, "drift_magnitude": "none"}

        # Calculate Jensen-Shannon divergence
        js_div = jensenshannon(self.previous_distribution, current_dist)

        drift_detected = js_div > self.threshold

        # Categorize drift magnitude
        if js_div < 0.05:
            magnitude = "none"
        elif js_div < 0.10:
            magnitude = "low"
        elif js_div < 0.20:
            magnitude = "medium"
        else:
            magnitude = "high"

        if drift_detected:
            drift_event = {
                "window_id": self.window_count,
                "timestamp": datetime.utcnow().isoformat(),
                "js_divergence": round(float(js_div), 4),
                "magnitude": magnitude,
                "previous_dist": [round(float(p), 4) for p in self.previous_distribution],
                "current_dist": [round(float(p), 4) for p in current_dist]
            }
            self.drift_events.append(drift_event)
            logger.warning(f"DRIFT DETECTED in window {self.window_count}: "
                          f"JS divergence = {js_div:.4f} ({magnitude})")

        self.previous_distribution = current_dist

        return {
            "drift_detected": drift_detected,
            "js_divergence": round(float(js_div), 4),
            "drift_magnitude": magnitude
        }

    def _start_new_window(self):
        """Start a new time window."""
        self.current_window = []
        self.window_count += 1

    def get_drift_summary(self) -> dict:
        """Get summary of all drift events."""
        return {
            "total_windows": self.window_count,
            "drift_events_count": len(self.drift_events),
            "drift_events": self.drift_events[-10:]  # Last 10 events
        }


def main():
    logger.info("Starting modeling service (LDA topic modeling + drift detection)...")
    logger.info(f"Configuration: {NUM_TOPICS} topics, batch size {BATCH_SIZE}")
    logger.info(f"Drift detection: window size {DRIFT_WINDOW_SIZE}, threshold {DRIFT_THRESHOLD}")

    consumer = create_consumer(KAFKA_TOPICS["normalized"], "modeling-group")
    producer = create_producer()
    modeler = TopicModeler(num_topics=NUM_TOPICS)
    drift_detector = DriftDetector(num_topics=NUM_TOPICS)

    # Buffer for batch processing
    message_buffer = deque()
    processed_count = 0

    logger.info(f"Consuming from {KAFKA_TOPICS['normalized']}, producing to {KAFKA_TOPICS['modeled']}")

    try:
        for message in consumer:
            try:
                msg = message.value
                text = msg.get("text", "")

                # Add to buffer
                message_buffer.append(msg)

                # When buffer is full, fit model and process batch
                if len(message_buffer) >= BATCH_SIZE:
                    # Extract texts for model fitting
                    texts = [m.get("text", "") for m in message_buffer]
                    modeler.fit_batch(texts)

                    # Process and produce each message
                    while message_buffer:
                        buffered_msg = message_buffer.popleft()
                        topics = modeler.get_topics(buffered_msg.get("text", ""))
                        drift_info = drift_detector.add_topic(topics["dominant_topic"])

                        output = {
                            "id": buffered_msg.get("id"),
                            "title": buffered_msg.get("title", ""),
                            "text": buffered_msg.get("text", ""),
                            "score": buffered_msg.get("score", 0.0),
                            "original_length": buffered_msg.get("original_length", 0),
                            "normalized_length": buffered_msg.get("normalized_length", 0),
                            "model_version": MODEL_VERSION,
                            "dominant_topic": topics["dominant_topic"],
                            "topic_distribution": topics["topic_distribution"],
                            "top_words": topics["top_words"],
                            "window_id": drift_info["window_id"],
                            "drift_detected": drift_info["drift_detected"],
                            "js_divergence": drift_info.get("js_divergence", 0.0),
                            "drift_magnitude": drift_info.get("drift_magnitude", "none")
                        }

                        producer.send(KAFKA_TOPICS["modeled"], value=output)
                        processed_count += 1

                    producer.flush()
                    logger.info(f"Processed {processed_count} messages")

                    # Log topic and drift summary periodically
                    if processed_count % 500 == 0:
                        summary = modeler.get_all_topics_summary()
                        for topic in summary[:3]:  # Log first 3 topics
                            logger.info(f"Topic {topic['topic_id']}: {', '.join(topic['top_words'][:5])}")
                        drift_summary = drift_detector.get_drift_summary()
                        logger.info(f"Drift summary: {drift_summary['drift_events_count']} events in {drift_summary['total_windows']} windows")

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue

    except KeyboardInterrupt:
        logger.info("Modeling service interrupted")
    finally:
        # Process remaining messages in buffer
        if message_buffer:
            logger.info(f"Processing {len(message_buffer)} remaining messages...")
            texts = [m.get("text", "") for m in message_buffer]
            if modeler.is_fitted or len(texts) >= 5:
                modeler.fit_batch(texts)

            while message_buffer:
                buffered_msg = message_buffer.popleft()
                topics = modeler.get_topics(buffered_msg.get("text", ""))
                drift_info = drift_detector.add_topic(topics["dominant_topic"])

                output = {
                    "id": buffered_msg.get("id"),
                    "title": buffered_msg.get("title", ""),
                    "text": buffered_msg.get("text", ""),
                    "score": buffered_msg.get("score", 0.0),
                    "original_length": buffered_msg.get("original_length", 0),
                    "normalized_length": buffered_msg.get("normalized_length", 0),
                    "model_version": MODEL_VERSION,
                    "dominant_topic": topics["dominant_topic"],
                    "topic_distribution": topics["topic_distribution"],
                    "top_words": topics["top_words"],
                    "window_id": drift_info["window_id"],
                    "drift_detected": drift_info["drift_detected"],
                    "js_divergence": drift_info.get("js_divergence", 0.0),
                    "drift_magnitude": drift_info.get("drift_magnitude", "none")
                }

                producer.send(KAFKA_TOPICS["modeled"], value=output)
                processed_count += 1

        # Log final drift summary
        drift_summary = drift_detector.get_drift_summary()
        logger.info(f"Final drift summary: {drift_summary['drift_events_count']} drift events detected")
        for event in drift_summary["drift_events"]:
            logger.info(f"  Window {event['window_id']}: JS={event['js_divergence']} ({event['magnitude']})")

        producer.flush()
        producer.close()
        consumer.close()
        logger.info(f"Modeling service shut down. Total processed: {processed_count}")


if __name__ == "__main__":
    main()
