"""Data ingestion module.

This module handles the loading of raw data from external sources (CSV for prices,
Parquet for blockchain metrics). It performs initial cleaning, schema validation,
and type casting to ensure downstream processing receives consistent data formats.
"""

import logging
from typing import Any, Dict, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampType

# Logger standard pour suivre le chargement
logger = logging.getLogger(__name__)


def load_price_data(spark: SparkSession, cfg: Dict[str, Any]) -> DataFrame:
    """Load historical price candles from CSV and normalize schema.

    The function filters data based on the configured date range, casts
    financial columns to DoubleType, and renames columns to a standard
    internal schema (open, high, low, close, volume).

    Args:
        spark: Active SparkSession.
        cfg: Global configuration dictionary.

    Returns:
        DataFrame with columns: [ts_raw, open, high, low, close, volume]
    """
    data_cfg = cfg["data"]
    path = data_cfg["price_path"]
    
    logger.info(f"Loading price data from: {path}")

    # Lecture CSV avec en-tête
    df = spark.read.option("header", True).csv(path)

    # 1. Gestion du Timestamp
    ts_col_name = data_cfg.get("price_timestamp_col", "Open time")
    # On trim les espaces et on convertit en timestamp
    df = df.withColumn(
        "ts_raw",
        F.to_timestamp(F.trim(F.col(ts_col_name)))
    )

    # 2. Filtrage temporel (Start / End date)
    start_date = data_cfg.get("start_date")
    end_date = data_cfg.get("end_date")

    if start_date:
        logger.info(f"Filtering prices >= {start_date}")
        df = df.filter(F.col("ts_raw") >= F.lit(start_date))
    
    if end_date:
        logger.info(f"Filtering prices <= {end_date}")
        df = df.filter(F.col("ts_raw") <= F.lit(end_date))

    # 3. Renommage et Cast (Transtypage)
    # Important : On force le type Double pour éviter les erreurs ML plus tard
    col_mapping = {
        data_cfg["price_open_col"]: "open",
        data_cfg["price_high_col"]: "high",
        data_cfg["price_low_col"]: "low",
        data_cfg["price_close_col"]: "close",
        data_cfg["price_volume_col"]: "volume",
    }

    for old_name, new_name in col_mapping.items():
        # On prend la colonne, on la cast en Double, et on la renomme
        df = df.withColumn(new_name, F.col(old_name).cast(DoubleType()))

    # Sélection finale propre
    return df.select("ts_raw", "open", "high", "low", "close", "volume")


def load_blockchain_data(spark: SparkSession, cfg: Dict[str, Any]) -> Optional[DataFrame]:
    """Load blockchain features from Parquet and align timestamps.

    This function attempts to detect the timestamp column dynamically
    to support different dataset sources (e.g., Yahoo Finance vs standard blocks).

    Args:
        spark: Active SparkSession.
        cfg: Global configuration dictionary.

    Returns:
        DataFrame with an hourly-truncated timestamp 'ts_hour' or None if disabled.
    """
    data_cfg = cfg.get("data", {})
    path = data_cfg.get("blockchain_path")

    # Vérification si la config demande la blockchain
    if not path:
        logger.warning("No blockchain path configured. Skipping blockchain features.")
        return None

    logger.info(f"Loading blockchain data from: {path}")
    
    try:
        df = spark.read.parquet(path)
    except Exception as e:
        logger.error(f"Failed to read parquet file at {path}: {e}")
        raise e

    cols = df.columns
    
    # 1. Détection intelligente de la colonne temporelle
    # On cherche 'ts_hour', 'timestamp', 'date' etc. insensible à la casse
    lower_map = {c.lower(): c for c in cols}
    candidates = ["ts_hour", "timestamp", "ts", "date", "datetime"]
    
    found_ts_col = None
    for cand in candidates:
        if cand in lower_map:
            found_ts_col = lower_map[cand]
            break

    if not found_ts_col:
        msg = f"No timestamp column found in blockchain data. Available columns: {cols}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Detected blockchain timestamp column: '{found_ts_col}'")

    # 2. Normalisation temporelle (Arrondi à l'heure)
    # Cela permet de faire la jointure parfaite avec les prix horaires
    df = df.withColumn(
        "ts_hour", 
        F.date_trunc("hour", F.col(found_ts_col).cast(TimestampType()))
    )

    return df