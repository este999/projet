"""Entry point for the Bitcoin price direction prediction pipeline.

This script orchestrates the end-to-end process:
1. Loading configuration.
2. Initializing the Spark session.
3. Ingesting raw datasets (Prices + Blockchain).
4. Feature engineering (Resampling, Indicators, Labeling).
5. Model training and evaluation.
6. Saving results and artifacts.

Usage:
    spark-submit src/main.py --config bda_project_config.yml
"""

import argparse
import logging
import sys
import time

from config import load_config
from spark_utils import create_spark
from ingestion import load_price_data, load_blockchain_data
from features import build_all_features
from modeling import build_train_test, train_and_evaluate


def setup_logging() -> logging.Logger:
    """Configure the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Réduire le bruit des bibliothèques tierces (py4j, etc.)
    logging.getLogger("py4j").setLevel(logging.ERROR)
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the BTC prediction pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="bda_project_config.yml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the pipeline logic."""
    logger = setup_logging()
    
    start_time = time.time()
    logger.info("Starting Bitcoin Prediction Pipeline...")

    args = parse_args()
    
    # Initialisation de variables pour le bloc finally
    spark = None

    try:
        # 1. Chargement de la Configuration
        cfg = load_config(args.config)
        logger.info("Configuration loaded successfully.")

        # 2. Initialisation Spark
        spark = create_spark("bda-btc-price-prediction", cfg)

        # 3. Ingestion des Données
        logger.info("--- Phase 1: Ingestion ---")
        df_prices = load_price_data(spark, cfg)
        df_blockchain = load_blockchain_data(spark, cfg)

        # 4. Feature Engineering (Orchestré par build_all_features)
        logger.info("--- Phase 2: Feature Engineering ---")
        # Cette fonction gère : resampling, indicateurs tech, jointure blockchain, labeling
        df_final = build_all_features(df_prices, df_blockchain, cfg)
        
        # Petit check debug dans les logs (sans spammer la console)
        logger.info(f"Feature Vector Columns: {df_final.columns}")

        # 5. Split Train / Test
        logger.info("--- Phase 3: Train/Test Split ---")
        train_df, test_df = build_train_test(df_final, cfg)

        # 6. Modélisation & Évaluation
        logger.info("--- Phase 4: Modeling ---")
        metrics = train_and_evaluate(train_df, test_df, cfg)

        # 7. Conclusion
        duration = time.time() - start_time
        logger.info("-" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total Duration: {duration:.2f} seconds")
        logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Final AUC:      {metrics['auc_roc']:.4f}")
        logger.info("-" * 50)

    except Exception as e:
        logger.critical(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)
        
    finally:
        if spark:
            logger.info("Stopping Spark Session...")
            spark.stop()


if __name__ == "__main__":
    main()