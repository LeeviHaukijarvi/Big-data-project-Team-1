"""
Modeling service - Kafka consumer that batches messages and runs LDA via run_modeling.
Includes drift detection to track topic distribution changes over time.
"""
import os
import sys
sys.path.insert(0, "/app")

import logging
import tempfile
import json
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.distance import jensenshannon

from shared.config import KAFKA_BROKER, KAFKA_TOPICS, BATCH_SIZE as DEFAULT_BATCH_SIZE
from shared.kafka_utils import create_consumer, create_producer
from run_modeling import run_modeling

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
BATCH_SIZE = int(os.environ.get("MODEL_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "pyspark-lda-v1")
NUM_TOPICS = int(os.environ.get("NUM_TOPICS", "10"))
DRIFT_WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "200"))
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.15"))


class DriftDetector:
    """Detects topic drift between time windows using Jensen-Shannon divergence."""

    def __init__(self, num_topics: int, window_size: int, threshold: float):
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
            return {"drift_detected": False, "window_id": self.window_count,
                    "js_divergence": 0.0, "drift_magnitude": "none"}

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
        dist = (counts + 1e-10) / (len(topics) + self.num_topics * 1e-10)
        return dist

    def _check_drift(self) -> dict:
        """Check for drift between current and previous window."""
        current_dist = self._get_distribution(self.current_window)

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
            drift_event = {
                "window_id": self.window_count,
                "timestamp": datetime.utcnow().isoformat(),
                "js_divergence": round(float(js_div), 4),
                "magnitude": magnitude
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
            "drift_events": self.drift_events[-10:]
        }


def process_batch(messages: list[dict], batch_num: int) -> list[dict]:
    """
    Process a batch of messages through PySpark LDA.

    Args:
        messages: List of message dictionaries with 'text' field
        batch_num: Batch number for logging

    Returns:
        List of enriched messages with topic assignments
    """
    logger.info(f"Processing batch {batch_num} with {len(messages)} messages")

    # Create DataFrame and save as parquet for PySpark
    df = pd.DataFrame(messages)

    # Tokenize text for LDA (simple whitespace tokenization)
    df['tokens'] = df['text'].apply(lambda x: x.split() if x else [])

    # Write to temp parquet file
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.parquet")
        output_path = os.path.join(tmpdir, "output")

        # Save with tokens column
        table = pa.Table.from_pandas(df)
        pq.write_table(table, input_path)

        try:
            # Run PySpark LDA
            topics_df = run_modeling(input_path, output_path)

            # Read results if available
            if topics_df is not None and len(topics_df) > 0:
                # Merge topic assignments back to messages
                for i, msg in enumerate(messages):
                    if i < len(topics_df):
                        row = topics_df.iloc[i]
                        msg['dominant_topic'] = int(row.get('dominant_topic', -1))
                        msg['topic_distribution'] = row.get('topic_distribution', [])
                        msg['top_words'] = row.get('top_words', [])
                    else:
                        msg['dominant_topic'] = -1
                        msg['topic_distribution'] = []
                        msg['top_words'] = []
                    msg['model_version'] = MODEL_VERSION
            else:
                # Fallback if PySpark fails
                for msg in messages:
                    msg['dominant_topic'] = -1
                    msg['topic_distribution'] = []
                    msg['top_words'] = []
                    msg['model_version'] = MODEL_VERSION

        except Exception as e:
            logger.error(f"PySpark modeling failed: {e}")
            # Add placeholder topic data
            for msg in messages:
                msg['dominant_topic'] = -1
                msg['topic_distribution'] = []
                msg['top_words'] = []
                msg['model_version'] = MODEL_VERSION

    return messages


def main():
    logger.info("Starting modeling service (PySpark LDA + Drift Detection)...")
    logger.info(f"Configuration: batch size {BATCH_SIZE}, {NUM_TOPICS} topics")
    logger.info(f"Drift detection: window {DRIFT_WINDOW_SIZE}, threshold {DRIFT_THRESHOLD}")

    consumer = create_consumer(KAFKA_TOPICS["normalized"], "modeling-group")
    producer = create_producer()
    drift_detector = DriftDetector(NUM_TOPICS, DRIFT_WINDOW_SIZE, DRIFT_THRESHOLD)

    message_buffer = deque()
    processed_count = 0
    batch_num = 0

    logger.info(f"Consuming from {KAFKA_TOPICS['normalized']}, producing to {KAFKA_TOPICS['modeled']}")

    try:
        for message in consumer:
            try:
                msg = message.value
                message_buffer.append(msg)

                # Process when buffer is full
                if len(message_buffer) >= BATCH_SIZE:
                    batch_num += 1
                    batch_messages = [message_buffer.popleft() for _ in range(BATCH_SIZE)]

                    # Run LDA on batch
                    enriched_messages = process_batch(batch_messages, batch_num)

                    # Add drift detection and produce enriched messages
                    for enriched_msg in enriched_messages:
                        drift_info = drift_detector.add_topic(enriched_msg.get('dominant_topic', -1))
                        enriched_msg['window_id'] = drift_info['window_id']
                        enriched_msg['drift_detected'] = drift_info['drift_detected']
                        enriched_msg['js_divergence'] = drift_info['js_divergence']
                        enriched_msg['drift_magnitude'] = drift_info['drift_magnitude']

                        producer.send(KAFKA_TOPICS["modeled"], value=enriched_msg)
                        processed_count += 1

                    producer.flush()
                    logger.info(f"Processed {processed_count} messages total")

                    # Log drift summary periodically
                    if batch_num % 5 == 0:
                        summary = drift_detector.get_drift_summary()
                        logger.info(f"Drift summary: {summary['drift_events_count']} events in {summary['total_windows']} windows")

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue

    except KeyboardInterrupt:
        logger.info("Modeling service interrupted")
    finally:
        # Process remaining messages
        if message_buffer:
            batch_num += 1
            remaining = list(message_buffer)
            logger.info(f"Processing {len(remaining)} remaining messages...")

            enriched_messages = process_batch(remaining, batch_num)
            for enriched_msg in enriched_messages:
                drift_info = drift_detector.add_topic(enriched_msg.get('dominant_topic', -1))
                enriched_msg['window_id'] = drift_info['window_id']
                enriched_msg['drift_detected'] = drift_info['drift_detected']
                enriched_msg['js_divergence'] = drift_info['js_divergence']
                enriched_msg['drift_magnitude'] = drift_info['drift_magnitude']

                producer.send(KAFKA_TOPICS["modeled"], value=enriched_msg)
                processed_count += 1

        # Log final drift summary
        summary = drift_detector.get_drift_summary()
        logger.info(f"Final drift summary: {summary['drift_events_count']} drift events detected")
        for event in summary['drift_events']:
            logger.info(f"  Window {event['window_id']}: JS={event['js_divergence']} ({event['magnitude']})")

        producer.flush()
        producer.close()
        consumer.close()
        logger.info(f"Modeling service shut down. Total processed: {processed_count}")


if __name__ == "__main__":
    main()
