"""Feature engineering functions for price and blockchain datasets.

This module defines the transformations required to resample minute‑level
price candles to hourly bars, compute derived statistics (returns,
volatility, ratios), and optionally join blockchain features. It also
creates binary labels for next‑horizon price direction.
"""

from typing import Dict, Optional, Any

from pyspark.sql import DataFrame, Window, functions as F


def build_price_features(df_prices: DataFrame, cfg: Dict[str, Any]) -> DataFrame:
    freq = cfg["etl"]["resample_freq"]  # e.g. "1 hour"

    # 🔹 1. Cast des colonnes numériques en double
    df_cast = (
        df_prices
        .withColumn("open", F.col("open").cast("double"))
        .withColumn("high", F.col("high").cast("double"))
        .withColumn("low", F.col("low").cast("double"))
        .withColumn("close", F.col("close").cast("double"))
        .withColumn("volume", F.col("volume").cast("double"))
    )

    # 🔹 2. Fenêtre horaire
    df = df_cast.withColumn("ts_hour_window", F.window("ts_raw", freq))

    # 🔹 3. Agrégation à l’heure
    agg = (
        df.groupBy("ts_hour_window")
        .agg(
            F.first("open").alias("open_1h"),
            F.last("close").alias("close_1h"),
            F.max("high").alias("high_1h"),
            F.min("low").alias("low_1h"),
            F.sum("volume").alias("vol_1h"),
            F.stddev_pop("close").alias("volatility_1h"),
        )
        .select(
            F.col("ts_hour_window").start.alias("ts_hour"),
            "open_1h",
            "close_1h",
            "high_1h",
            "low_1h",
            "vol_1h",
            "volatility_1h",
        )
    )

    # 🔹 4. Ajout du retour 1h (log-return) et du ratio high/low
    w = Window.orderBy("ts_hour")
    agg = (
        agg
        .withColumn("close_prev", F.lag("close_1h", 1).over(w))  # offset = 1
        .withColumn("ret_1h", F.log(F.col("close_1h") / F.col("close_prev")))
        .withColumn("high_low_ratio", F.col("high_1h") / F.col("low_1h"))
        .drop("close_prev")
    )

    # On enlève les lignes où on ne peut pas calculer (première ligne, etc.)
    return agg.dropna()


def join_blockchain_features(df_price_feat: DataFrame, df_blockchain: Optional[DataFrame], cfg: Dict[str, any]) -> DataFrame:
    """Join blockchain features onto price features if available.

    Args:
        df_price_feat: DataFrame of price features with ``ts_hour`` column.
        df_blockchain: DataFrame of blockchain features or ``None``.
        cfg: Loaded configuration dictionary.

    Returns:
        A DataFrame containing price features and, if ``use_blockchain`` is
        ``True``, the blockchain features joined on timestamp.
    """
    if df_blockchain is None or not cfg["features"].get("use_blockchain", False):
        return df_price_feat
    return df_price_feat.join(df_blockchain, on="ts_hour", how="left")


def add_labels(df_features: DataFrame, cfg: Dict[str, any]) -> DataFrame:
    """Create binary labels indicating the direction of future price movement.

    The label ``label_up`` is ``1`` when the future return over the horizon
    specified in the config is positive, and ``0`` otherwise.

    Args:
        df_features: A DataFrame produced by
            :func:`build_price_features` or the equivalent.
        cfg: Loaded configuration dictionary containing the horizon in
            hours for the label.

    Returns:
        A DataFrame with ``return_future``, ``label_up`` and no null
        futures.
    """
    horizon_hours = cfg["etl"]["label_horizon_hours"]
    w = Window.orderBy("ts_hour")

    df = df_features.withColumn(
        "close_future", F.lead("close_1h", horizon_hours).over(w)
    )
    df = df.withColumn(
        "return_future",
        (F.col("close_future") - F.col("close_1h")) / F.col("close_1h")
    )
    df = df.withColumn(
        "label_up",
        F.when(F.col("return_future") > 0, F.lit(1)).otherwise(F.lit(0))
    )
    return df.dropna(subset=["close_future", "return_future"])