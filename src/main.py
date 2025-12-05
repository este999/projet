"""Entry point for the Bitcoin price direction prediction pipeline.

This script orchestrates the end‑to‑end process: loading configuration,
initialising Spark, ingesting datasets, engineering features, training a
model and saving results. Run with ``spark-submit`` and pass the path
to your YAML config using ``--config``.
"""

import argparse
from pathlib import Path

from config import load_config
from spark_utils import create_spark
from ingestion import load_price_data, load_blockchain_data
from features import build_price_features, join_blockchain_features, add_labels, build_all_features
from modeling import build_train_test, train_and_evaluate


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the pipeline runner."""
    parser = argparse.ArgumentParser(description="Run the BTC prediction pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="bda_project_config.yml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the pipeline based on the provided configuration."""
    args = parse_args()
    cfg_path = args.config
    cfg = load_config(cfg_path)

    # Create Spark session
    spark = create_spark("bda-btc-price-movement", cfg)

    # Load price data
    df_prices = load_price_data(spark, cfg)

    # Optionally load blockchain data
    df_blockchain = load_blockchain_data(spark, cfg)

    # Feature engineering
    df_price_feat = build_price_features(df_prices, cfg)
    df_feat = join_blockchain_features(df_price_feat, df_blockchain, cfg)
    df_labeled = add_labels(df_feat, cfg)

    print("Cols features :", df_feat.columns)

    df_feat.select(
    "ts_hour",
    "tx_count_hour",
    "total_value_hour"
    ).orderBy("ts_hour").show(5)

    # ✅ Train/test split sur le DF avec le label
    train_df, test_df = build_train_test(df_labeled, cfg)

    # Train model and evaluate
    metrics = train_and_evaluate(train_df, test_df, cfg)
    print("Metrics:", metrics)

    spark.stop()



if __name__ == "__main__":
    main()