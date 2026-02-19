"""
Modeling service entry point.
This module is called by the backend pipeline.
"""

def run_modeling(input_path: str = "data/normalized/"):
    print("Modeling started")
    print(f"Reading data from {input_path}")

    # Topic modeling logic will be implemented here
    # (e.g., Spark LDA / BERTopic)

    print("Modeling finished successfully")


if __name__ == "__main__":
    run_modeling()
