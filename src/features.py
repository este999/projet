"""Feature engineering functions for price and blockchain datasets.

This module defines the transformations required to resample minute‑level
price candles to hourly bars, compute derived statistics (returns,
volatility, ratios), and optionally join blockchain features. It also
creates binary labels for next‑horizon price direction.
"""

from typing import Dict, Optional, Any

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window


def build_price_features(df_prices: DataFrame, cfg: Dict[str, Any]) -> DataFrame:
    """
    Construit les features à partir des prix :
    - agrégation en bougies 1h
    - volatilité intra-heure
    - rendement futur 1h (ret_1h)
    - high_low_ratio
    - lags + moyennes mobiles + vol glissante
    """

    # 1) Agrégation en 1h
    hourly = (
        df_prices
        .withColumn("ts_hour_window", F.window("ts_raw", "1 hour"))
        .groupBy("ts_hour_window")
        .agg(
            F.first(F.col("open").cast("double")).alias("open_1h"),
            F.last(F.col("close").cast("double")).alias("close_1h"),
            F.max(F.col("high").cast("double")).alias("high_1h"),
            F.min(F.col("low").cast("double")).alias("low_1h"),
            F.sum(F.col("volume").cast("double")).alias("vol_1h"),
            F.stddev_pop(F.col("close").cast("double")).alias("volatility_1h"),
        )
        .select(
            F.col("ts_hour_window").getField("start").alias("ts_hour"),
            "open_1h",
            "close_1h",
            "high_1h",
            "low_1h",
            "vol_1h",
            "volatility_1h",
        )
    )

    # 2) Fenêtres temporelles
    w = Window.orderBy("ts_hour")
    w_3h = w.rowsBetween(-2, 0)
    w_12h = w.rowsBetween(-11, 0)
    w_24h = w.rowsBetween(-23, 0)

    # 3) Rendements + ratios + features dérivées
       # 3) Rendements + ratios + features dérivées
    feats = (
        hourly
        .withColumn("close_1h_d", F.col("close_1h").cast("double"))
        # prix *passé* (t-1h)
        .withColumn("close_prev", F.lag("close_1h_d", 1).over(w))
        # rendement log passé : ln(P_t / P_{t-1})
        .withColumn(
            "ret_1h",
            F.log(F.col("close_1h_d") / F.col("close_prev"))
        )
        # ratio high / low
        .withColumn("high_low_ratio", F.col("high_1h") / F.col("low_1h"))
        # lags du rendement (info encore plus ancienne)
        .withColumn("ret_1h_lag1", F.lag("ret_1h", 1).over(w))
        .withColumn("ret_1h_lag2", F.lag("ret_1h", 2).over(w))
        # moyennes mobiles de prix
        .withColumn("ma_close_3h", F.avg("close_1h_d").over(w_3h))
        .withColumn("ma_close_12h", F.avg("close_1h_d").over(w_12h))
        # volatilité glissante sur 24h
        .withColumn("vol_24h", F.stddev("close_1h_d").over(w_24h))
        .drop("close_1h_d", "close_prev")
        .orderBy("ts_hour")
    )


    return hourly



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

    # Seuil de 0.1 % pour considérer un "vrai" mouvement
    THRESH = 0.001  # 0.1%


    df = df.withColumn(
        "label_up",
        F.when(F.col("return_future") > THRESH, F.lit(1)).otherwise(F.lit(0))
    )
    return df.dropna(subset=["close_future", "return_future"])

def build_all_features(
    df_prices: DataFrame,
    df_blockchain: Optional[DataFrame],
    cfg: Dict[str, Any]
) -> DataFrame:
    """
    Construit les features de prix + (optionnel) blockchain,
    puis ajoute le label.
    """
    df_price_feat = build_price_features(df_prices, cfg)

    use_blockchain = cfg.get("features", {}).get("use_blockchain", False)

    if use_blockchain and df_blockchain is not None:
        df_feat = (
            df_price_feat.alias("p")
            .join(df_blockchain.alias("b"), on="ts_hour", how="left")
        )
    else:
        df_feat = df_price_feat

    df_labeled = build_label(df_feat, cfg)
    return df_labeled