"""
Normalization service - consumes raw text from Kafka, normalizes it, and produces to normalized-text topic.
Performs lowercase conversion, URL removal, HTML tag stripping, special character removal, and whitespace normalization.
"""
import sys
sys.path.insert(0, "/app")

from shared.config import KAFKA_BROKER, KAFKA_TOPICS
from shared.kafka_utils import create_consumer, create_producer
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text by applying cleaning transformations.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove special characters (keep basic punctuation)
    5. Collapse multiple whitespace

    Args:
        text: Raw text string

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?;:\'-]', '', text)

    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


if __name__ == "__main__":
    try:
        logger.info("Starting normalization service...")

        # Create consumer and producer
        consumer = create_consumer(KAFKA_TOPICS["raw"], "normalization-group")
        producer = create_producer()

        processed_count = 0

        logger.info(f"Consuming from {KAFKA_TOPICS['raw']}, producing to {KAFKA_TOPICS['normalized']}")

        for message in consumer:
            try:
                msg = message.value

                # Extract fields
                msg_id = msg.get("id", "unknown")
                raw_text = msg.get("text", "")
                raw_title = msg.get("title", "")

                # Normalize text and title
                normalized_text = normalize_text(raw_text)
                normalized_title = normalize_text(raw_title)

                # Create output message (pass through score and reddit_scores
                # from data_preparation for downstream use)
                output = {
                    "id": msg_id,
                    "title": normalized_title,
                    "text": normalized_text,
                    "original_length": len(raw_text),
                    "normalized_length": len(normalized_text),
                    "score": msg.get("score", 0.0),
                    "reddit_scores": msg.get("reddit_scores", []),
                }

                # Produce to normalized-text topic
                producer.send(KAFKA_TOPICS["normalized"], value=output)

                processed_count += 1

                # Log progress every 100 messages
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} messages")

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue

        logger.info(f"Normalization service shutting down. Processed {processed_count} messages total.")
        producer.flush()
        consumer.close()

    except KeyboardInterrupt:
        logger.info("Normalization service interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in normalization service: {e}", exc_info=True)
        raise
