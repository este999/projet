"""Spark session helper functions.

This module contains utilities for constructing a SparkSession with
project-specific configuration options. Centralising session creation
allows you to tweak configuration (memory settings, shuffle partitions,
logging levels) in one place without scattering constants throughout
your codebase.
"""

from typing import Any, Dict

from pyspark.sql import SparkSession


def create_spark(app_name: str, cfg: Dict[str, Any]) -> SparkSession:
    """Instantiate and configure a SparkSession.

    Reads configuration from the ``spark`` section of the YAML config.

    Args:
        app_name: The human‑friendly name for your Spark application.
        cfg: A dictionary loaded from YAML containing a ``spark`` section.

    Returns:
        A configured :class:`pyspark.sql.SparkSession` instance.
    """
    spark_cfg = cfg.get("spark", {})
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(spark_cfg.get("master", "local[*]"))
    )

    # Tune shuffle partitions to an appropriate value for local runs.
    builder = builder.config(
        "spark.sql.shuffle.partitions",
        str(spark_cfg.get("shuffle_partitions", 8)),
    )

    spark = builder.getOrCreate()

    # Set log level (WARN by default) to reduce verbosity.
    log_level = spark_cfg.get("log_level", "WARN")
    spark.sparkContext.setLogLevel(log_level)
    return spark