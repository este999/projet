from typing import Dict, Any, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def load_price_data(spark: SparkSession, cfg: Dict[str, Any]) -> DataFrame:
    """Load historical price candles from CSV into a canonical schema.

    Returns a DataFrame with:
    - ts_raw (timestamp)
    - open, high, low, close, volume (double)
    """
    data_cfg = cfg["data"]
    path = data_cfg["price_path"]

    # Lecture brute du CSV (toutes les colonnes arrivent en string)
    df = (
        spark.read
        .option("header", True)
        .csv(path)
    )

    ts_col = data_cfg["price_timestamp_col"]
    # True si la colonne est un Unix timestamp en secondes (ex: "1562352720.0")
    ts_is_unix = data_cfg.get("timestamp_is_unix", True)

    if ts_is_unix:
        # "1453430580.0" -> cast en double -> from_unixtime -> timestamp
        df = df.withColumn(
            "ts_raw",
            F.from_unixtime(F.col(ts_col).cast("double")).cast("timestamp")
        )
    else:
        # Cas où la colonne est déjà une string de type "YYYY-MM-DD HH:MM:SS"
        df = df.withColumn("ts_raw", F.to_timestamp(F.col(ts_col)))

    # Filtre sur la période
    start = data_cfg.get("start_date")
    end = data_cfg.get("end_date")
    if start:
        df = df.filter(F.col("ts_raw") >= F.lit(start))
    if end:
        df = df.filter(F.col("ts_raw") <= F.lit(end))

    # Renommage en schéma canonique
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


def load_blockchain_data(spark: SparkSession, cfg: Dict[str, Any]) -> Optional[DataFrame]:
    """Load blockchain-derived features if a path is provided.

    Returns None when no blockchain dataset is configured.
    """
    data_cfg = cfg.get("data", {})
    path = data_cfg.get("blockchain_path")

    # Si tu n'as pas encore de dataset blockchain, laisse blockchain_path: null dans le YAML
    if not path:
        return None

    df = spark.read.parquet(path)
    return df
