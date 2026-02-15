"""
Reusable Kafka producer and consumer factory functions.
Includes retry logic for connection reliability in Docker environments.
"""
import logging
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable
import json

from shared.config import KAFKA_BROKER

logger = logging.getLogger(__name__)


def create_producer(broker=None):
    """
    Create a Kafka producer with JSON serialization.

    Args:
        broker: Kafka broker address (defaults to config.KAFKA_BROKER)

    Returns:
        KafkaProducer instance

    Retries up to 10 times with 5-second delays to handle Kafka startup delays.
    """
    if broker is None:
        broker = KAFKA_BROKER

    max_retries = 10
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempting to connect to Kafka broker at {broker} (attempt {attempt}/{max_retries})")
            producer = KafkaProducer(
                bootstrap_servers=broker,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info(f"Successfully connected to Kafka broker at {broker}")
            return producer
        except NoBrokersAvailable as e:
            if attempt < max_retries:
                logger.warning(f"Failed to connect to Kafka broker (attempt {attempt}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to Kafka broker after {max_retries} attempts")
                raise


def create_consumer(topic, group_id, broker=None):
    """
    Create a Kafka consumer with JSON deserialization.

    Args:
        topic: Topic name or list of topic names to subscribe to
        group_id: Consumer group identifier
        broker: Kafka broker address (defaults to config.KAFKA_BROKER)

    Returns:
        KafkaConsumer instance

    Retries up to 10 times with 5-second delays to handle Kafka startup delays.
    """
    if broker is None:
        broker = KAFKA_BROKER

    max_retries = 10
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempting to connect to Kafka broker at {broker} for topic {topic} (attempt {attempt}/{max_retries})")
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=broker,
                group_id=group_id,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            logger.info(f"Successfully connected to Kafka broker at {broker} for topic {topic}")
            return consumer
        except NoBrokersAvailable as e:
            if attempt < max_retries:
                logger.warning(f"Failed to connect to Kafka broker (attempt {attempt}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to Kafka broker after {max_retries} attempts")
                raise
