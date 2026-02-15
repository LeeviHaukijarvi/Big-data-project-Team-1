# Phase 1 stub: passthrough consumer. Real PySpark topic modeling implemented in Phase 2.
"""
Modeling service - consumes normalized text from Kafka and produces to modeled-text topic.

This is a Phase 1 stub that passes messages through with stub model metadata.
Real PySpark topic modeling (LDA) will be implemented in Phase 2.
"""
import sys
sys.path.insert(0, "/app")

from shared.config import KAFKA_BROKER, KAFKA_TOPICS
from shared.kafka_utils import create_consumer, create_producer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    try:
        logger.info("Starting modeling service (Phase 1 stub)...")

        # Create consumer and producer
        consumer = create_consumer(KAFKA_TOPICS["normalized"], "modeling-group")
        producer = create_producer()

        processed_count = 0

        logger.info(f"Consuming from {KAFKA_TOPICS['normalized']}, producing to {KAFKA_TOPICS['modeled']}")

        for message in consumer:
            try:
                msg = message.value

                # Add stub model metadata
                output = {
                    **msg,  # Include all fields from input
                    "model_version": "stub-v0",
                    "topics": []  # Empty topics list - placeholder for Phase 2
                }

                # Produce to modeled-text topic
                producer.send(KAFKA_TOPICS["modeled"], value=output)

                processed_count += 1

                # Log progress every 100 messages
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} messages (passthrough)")

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue

        logger.info(f"Modeling service shutting down. Processed {processed_count} messages total.")
        producer.flush()
        consumer.close()

    except KeyboardInterrupt:
        logger.info("Modeling service interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in modeling service: {e}", exc_info=True)
        raise
