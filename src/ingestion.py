"""Data ingestion functions for the Bitcoin price prediction pipeline.

This module defines helper functions to load price data and blockchain
data into PySpark DataFrames. The price ingestion supports standard
CSV formats downloaded from Kaggle, while the blockchain ingestion is
implemented as a stub that can be extended once a decoded blockchain
dataset (e.g. Parquet) becomes available.
"""

from typing import Dict, Optional

from pyspark.sql import DataFrame, SparkSession, functions as F


def load_price_data(spark: SparkSession, cfg: Dict[str, any]) -> DataFrame:
    """Load historical price candles from CSV.

    The data is filtered to the date range specified in the config and
    columns are renamed to a standard schema (open, high, low, close,
    volume).

    Args:
        spark: Active Spark session.
        cfg: Loaded configuration dictionary.

    Returns:
        A Spark DataFrame with a timestamp column named ``ts_raw`` and
        columns ``open``, ``high``, ``low``, ``close`` and ``volume``.
    """
    data_cfg = cfg["data"]
    path = data_cfg["price_path"]

    # Read raw CSV with header. Infers column types as strings.
    df = (
        spark.read
        .option("header", True)
        .csv(path)
    )

    # Cast the timestamp column to actual timestamps. Column names can vary
    # between Kaggle datasets (e.g. "Timestamp" or "date").
    ts_col = data_cfg["price_timestamp_col"]
    df = df.withColumn("ts_raw", F.col(ts_col).cast("timestamp"))

    # Filter the date range as defined in YAML.
    start = data_cfg.get("start_date")
    end = data_cfg.get("end_date")
    if start:
        df = df.filter(F.col("ts_raw") >= F.lit(start))
    if end:
        df = df.filter(F.col("ts_raw") <= F.lit(end))

    # Rename columns to a canonical schema.
    df = (
        df
        .withColumnRenamed(data_cfg["price_open_col"], "open")
        .withColumnRenamed(data_cfg["price_high_col"], "high")
        .withColumnRenamed(data_cfg["price_low_col"], "low")
        .withColumnRenamed(data_cfg["price_close_col"], "close")
        .withColumnRenamed(data_cfg["price_volume_col"], "volume")
        .select("ts_raw", "open", "high", "low", "close", "volume")
    )
    return df


def load_blockchain_data(spark: SparkSession, cfg: Dict[str, any]) -> Optional[DataFrame]:
    """Load blockchain features from Parquet (placeholder).

    This function returns ``None`` when no ``blockchain_path`` is
    configured. When a Parquet dataset is available, it should return
    a DataFrame with at least a timestamp column (timestamp truncated
    to the hour) and aggregated blockchain metrics such as transaction
    counts or total transferred value.

    Args:
        spark: Active Spark session.
        cfg: Loaded configuration dictionary.

    Returns:
        A DataFrame with blockchain features or ``None`` if absent.
    """
    path = cfg["data"].get("blockchain_path")
    if not path:
        # No blockchain dataset configured. Baseline will run on price data.
        return None

    # Load Parquet dataset. Assumes schema is already appropriate.
    df = spark.read.parquet(path)
    return df