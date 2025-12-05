from typing import Dict, Any, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def load_price_data(spark: SparkSession, cfg: Dict[str, Any]) -> DataFrame:
    """Load historical price candles from CSV.

    The data is filtered to the date range specified in the config and
    columns are renamed to a standard schema (open, high, low, close,
    volume).

    Returns:
        A Spark DataFrame with a timestamp column named ``ts_raw`` and
        columns ``open``, ``high``, ``low``, ``close`` and ``volume``.
    """
    data_cfg = cfg["data"]
    path = data_cfg["price_path"]

    # Read raw CSV with header. Types are strings by default.
    df = (
        spark.read
        .option("header", True)
        .csv(path)
    )

    # Colonne de timestamp (dans ton YAML: "Open time")
    ts_col = data_cfg["price_timestamp_col"]

    # Ici : on considère que la colonne contient déjà une date texte
    # style "2018-01-01 00:00:00.000000 ".
    # -> on trim et on parse directement en timestamp.
    df = df.withColumn(
        "ts_raw",
        F.to_timestamp(F.trim(F.col(ts_col)))
    )

    # Filtre sur la plage de dates si défini dans le YAML
    start = data_cfg.get("start_date")
    end = data_cfg.get("end_date")
    if start:
        df = df.filter(F.col("ts_raw") >= F.lit(start))
    if end:
        df = df.filter(F.col("ts_raw") <= F.lit(end))

    # Canonicalisation des colonnes de prix
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
    """
    Charge les features blockchain horaires depuis un parquet
    et crée une colonne ts_hour alignée avec les prix.
    """
    data_cfg = cfg.get("data", {})
    path = data_cfg.get("blockchain_path")

    # Si pas de chemin configuré → pas de features blockchain
    if not path:
        print("[INFO] Pas de blockchain_path dans la config → on saute la partie blockchain.")
        return None

    print(f"[INFO] Lecture des features blockchain depuis : {path}")
    df = spark.read.parquet(path)

    cols = df.columns
    print(f"[INFO] Colonnes blockchain lues par Spark : {cols}")

    # 1) Recherche de la colonne temporelle
    #    On fait une détection *insensible à la casse* et on inclut 'timestamp'
    lower_map = {c.lower(): c for c in cols}
    candidates = ["ts_hour", "timestamp", "ts", "time", "block_time", "block_timestamp"]

    ts_col = None
    for cand in candidates:
        if cand in lower_map:
            ts_col = lower_map[cand]
            break

    if ts_col is None:
        # Si vraiment rien trouvé, on lève une erreur explicite
        raise ValueError(
            f"Impossible de trouver une colonne temporelle dans le parquet blockchain. "
            f"Colonnes trouvées : {cols}"
        )

    print(f"[INFO] Colonne temporelle détectée pour la blockchain : {ts_col}")

    # 2) On normalise en timestamp Spark et on tronque à l'heure
    df = df.withColumn("ts_hour", F.date_trunc("hour", F.col(ts_col).cast("timestamp")))

    # 3) On met ts_hour en premier pour le join avec les features prix
    other_cols = [c for c in cols if c != ts_col]
    ordered_cols = ["ts_hour"] + other_cols
    df = df.select(*ordered_cols)

    return df
