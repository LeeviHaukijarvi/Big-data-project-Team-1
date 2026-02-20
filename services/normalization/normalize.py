"""
Normalization service - consumes raw text from Kafka, normalizes it, and produces to normalized-text topic.
Performs lowercase conversion, URL removal, HTML tag stripping, special character removal, and whitespace normalization.
"""
import sys
import re
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.insert(0, "/app")

from shared.config import KAFKA_BROKER, KAFKA_TOPICS, BATCH_SIZE
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

    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove time
    text = re.sub(r'\b\d{1,2}:\d{2}:\d{2}\s*(am|pm)?\b', '', text)

    # Remove dates
    text = re.sub(
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december|'
    r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
    r'\s+\d{1,2}(,\s*\d{4})?\b',
    '',
    text,
    flags=re.IGNORECASE)
    
    # Remove page markers like [ 9 ]
    text = re.sub(r'\[\s*\d+\s*\]', '', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # removing nos..
    text = re.sub(r'\d+', '', text)

    # Remove dollar amounts
    text = re.sub(r'\$\s?\d[\d,]*', '', text)

    # Remove repeated punctuation
    text = re.sub(r'([!?.,])\1+', r'\1', text)

    text = text.replace('-', ' ')

    # Remove standalone years
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)

    stop_words = set(ENGLISH_STOP_WORDS)

    def remove_stopwords(text):
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        return " ".join(tokens)
    
    text = remove_stopwords(text)

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

                # Log progress every BATCH_SIZE messages
                if processed_count % BATCH_SIZE == 0:
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
