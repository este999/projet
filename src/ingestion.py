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


def load_blockchain_data(
    spark: SparkSession,
    cfg: Dict[str, Any]
) -> Optional[DataFrame]:
    """
    Charge les métriques on-chain (ex. tx_count, volume on-chain) et
    les agrège à l'heure pour matcher les bougies prix.
    """
    data_cfg = cfg["data"]
    path = data_cfg.get("blockchain_path")
    if not path:
        return None

    # Lecture générique CSV/parquet
    if path.lower().endswith(".parquet"):
        df = spark.read.parquet(path)
    else:
        df = (
            spark.read
            .option("header", True)
            .csv(path)
        )

    # Colonne timestamp de la blockchain
    ts_col = data_cfg.get("blockchain_timestamp_col", "timestamp")

    # Parsing du timestamp (unix ou datetime)
    if data_cfg.get("timestamp_is_unix", False):
        df = df.withColumn(
            "ts_raw",
            F.from_unixtime(F.col(ts_col).cast("double"), "yyyy-MM-dd HH:mm:ss").cast("timestamp")
        )
    else:
        df = df.withColumn("ts_raw", F.col(ts_col).cast("timestamp"))

    # Filtre sur la même période que les prix
    start = data_cfg.get("start_date")
    end = data_cfg.get("end_date")
    if start:
        df = df.filter(F.col("ts_raw") >= F.lit(start))
    if end:
        df = df.filter(F.col("ts_raw") <= F.lit(end))

    # Agrégation horaire
    df = df.withColumn("ts_hour", F.date_trunc("hour", F.col("ts_raw")))

    metric_cols = [c for c in df.columns if c not in {ts_col, "ts_raw", "ts_hour"}]

    df_hourly = (
        df.groupBy("ts_hour")
          .agg(*[F.avg(c).alias(c) for c in metric_cols])
          .orderBy("ts_hour")
    )

    return df_hourly
