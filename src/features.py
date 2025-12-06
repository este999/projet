"""Feature engineering module.

This module defines the transformations required to convert raw price and
blockchain data into a feature set suitable for machine learning.
It handles:
1. Resampling minute-level data to hourly bars.
2. Computing technical indicators (returns, volatility, ratios).
3. Joining blockchain metrics.
4. Generating target labels (future price direction).
"""

import logging
from typing import Dict, Optional, Any

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


def build_price_features(df_prices: DataFrame, cfg: Dict[str, Any]) -> DataFrame:
    """Aggregate raw prices to hourly candles and compute technical features.

    Features computed:
    - OHLCV (Open, High, Low, Close, Volume) aggregated by hour.
    - Volatility (stddev of close within the hour).
    - Returns (1h).
    - High/Low Ratio.
    - Moving averages (3h, 12h).
    - Rolling volatility (24h).

    Args:
        df_prices: DataFrame with 'ts_raw' and raw price columns.
        cfg: Global configuration dictionary.

    Returns:
        DataFrame with aggregated hourly data and derived features.
    """
    logger.info("Building price features...")

    # 1. Agrégation horaire (Resampling)
    # On crée une fenêtre temporelle fixe de 1h
    hourly = (
        df_prices
        .withColumn("ts_hour_window", F.window("ts_raw", "1 hour"))
        .groupBy("ts_hour_window")
        .agg(
            F.first(F.col("open").cast(DoubleType())).alias("open_1h"),
            F.last(F.col("close").cast(DoubleType())).alias("close_1h"),
            F.max(F.col("high").cast(DoubleType())).alias("high_1h"),
            F.min(F.col("low").cast(DoubleType())).alias("low_1h"),
            F.sum(F.col("volume").cast(DoubleType())).alias("vol_1h"),
            F.stddev_pop(F.col("close").cast(DoubleType())).alias("volatility_1h"),
        )
        .select(
            F.col("ts_hour_window").getField("start").alias("ts_hour"),
            "open_1h", "close_1h", "high_1h", "low_1h", "vol_1h", "volatility_1h"
        )
    )

    # 2. Définition des fenêtres glissantes (Window Specs)
    w = Window.orderBy("ts_hour")
    w_3h = w.rowsBetween(-2, 0)     # Fenêtre de 3h (inclus l'heure courante)
    w_12h = w.rowsBetween(-11, 0)   # Fenêtre de 12h
    w_24h = w.rowsBetween(-23, 0)   # Fenêtre de 24h

    # 3. Calcul des indicateurs techniques
    feats = (
        hourly
        .withColumn("close_1h_d", F.col("close_1h"))  # Helper pour calculs
        # Prix précédent (lag 1)
        .withColumn("close_prev", F.lag("close_1h_d", 1).over(w))
        # Rendement Logarithmique : ln(Pt / Pt-1)
        .withColumn(
            "ret_1h",
            F.log(F.col("close_1h_d") / F.col("close_prev"))
        )
        # Ratio Volatilité / Amplitude
        .withColumn("high_low_ratio", F.col("high_1h") / F.col("low_1h"))
        # Retards (Lags) des rendements
        .withColumn("ret_1h_lag1", F.lag("ret_1h", 1).over(w))
        .withColumn("ret_1h_lag2", F.lag("ret_1h", 2).over(w))
        # Moyennes Mobiles Simples (SMA)
        .withColumn("ma_close_3h", F.avg("close_1h_d").over(w_3h))
        .withColumn("ma_close_12h", F.avg("close_1h_d").over(w_12h))
        # Volatilité glissante sur 24h
        .withColumn("vol_24h", F.stddev("close_1h_d").over(w_24h))
        # Nettoyage des colonnes temporaires
        .drop("close_1h_d", "close_prev")
        .orderBy("ts_hour")
    )

    # Correction critique : on retourne 'feats', pas 'hourly' !
    return feats


def join_blockchain_features(
    df_price_feat: DataFrame, 
    df_blockchain: Optional[DataFrame], 
    cfg: Dict[str, Any]
) -> DataFrame:
    """Join blockchain features onto price features if enabled in config.

    Args:
        df_price_feat: Hourly price features DataFrame.
        df_blockchain: Hourly blockchain features DataFrame (or None).
        cfg: Global configuration dictionary.

    Returns:
        DataFrame with joined features. Returns df_price_feat unchanged if
        blockchain is disabled or unavailable.
    """
    feat_cfg = cfg.get("features", {})
    use_blockchain = feat_cfg.get("use_blockchain", False)

    if df_blockchain is None or not use_blockchain:
        logger.info("Skipping blockchain join (disabled or no data).")
        return df_price_feat

    logger.info("Joining blockchain features...")
    # Jointure gauche pour ne pas perdre d'heures de prix si la blockchain a des trous
    return df_price_feat.join(df_blockchain, on="ts_hour", how="left")


def add_labels(df_features: DataFrame, cfg: Dict[str, Any]) -> DataFrame:
    """Create the target variable (Label) for prediction.

    The label is binary: 1 if the future return is positive (> threshold), 0 otherwise.

    Args:
        df_features: DataFrame containing features.
        cfg: Global configuration dictionary.

    Returns:
        DataFrame with new columns 'return_future' and 'label_up'.
        Rows with null labels (end of dataset) are dropped.
    """
    etl_cfg = cfg.get("etl", {})
    horizon_hours = etl_cfg.get("label_horizon_hours", 1)
    
    logger.info(f"Generating labels with horizon={horizon_hours}h...")

    w = Window.orderBy("ts_hour")

    # On regarde le prix dans le futur (Shift négatif / Lead)
    df = df_features.withColumn(
        "close_future", F.lead("close_1h", horizon_hours).over(w)
    )

    # Calcul du rendement futur
    df = df.withColumn(
        "return_future",
        (F.col("close_future") - F.col("close_1h")) / F.col("close_1h")
    )

    # Seuil pour considérer une hausse significative (évite le bruit autour de 0)
    # 0.001 = 0.1%
    threshold = 0.001 
    
    df = df.withColumn(
        "label_up",
        F.when(F.col("return_future") > threshold, F.lit(1)).otherwise(F.lit(0))
    )

    # On supprime les dernières lignes qui n'ont pas de futur (NaN)
    count_before = df.count()
    df_clean = df.dropna(subset=["close_future", "return_future"])
    
    # (Optionnel : logger combien de lignes sont perdues à la fin)
    # logger.debug(f"Dropped {count_before - df_clean.count()} rows (end of history).")

    return df_clean


def build_all_features(
    df_prices: DataFrame,
    df_blockchain: Optional[DataFrame],
    cfg: Dict[str, Any]
) -> DataFrame:
    """Orchestrator function to build the complete dataset.

    Pipeline:
    1. Build price features.
    2. Join blockchain features (optional).
    3. Add prediction labels.

    Args:
        df_prices: Raw price DataFrame.
        df_blockchain: Raw blockchain DataFrame.
        cfg: Configuration dictionary.

    Returns:
        Final DataFrame ready for training.
    """
    # 1. Features de prix
    df_price_feat = build_price_features(df_prices, cfg)

    # 2. Ajout Blockchain (utilise la fonction helper dédiée)
    df_merged = join_blockchain_features(df_price_feat, df_blockchain, cfg)

    # 3. Création des labels (Correction : appel de add_labels et non build_label)
    df_labeled = add_labels(df_merged, cfg)

    return df_labeled