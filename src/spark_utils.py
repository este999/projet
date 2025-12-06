"""Spark session management utilities.

This module handles the initialization and configuration of the Apache Spark
session. It centralizes Spark-specific settings (master node, shuffle partitions,
logging levels) derived from the project configuration file.
"""

import logging
from typing import Any, Dict

from pyspark.sql import SparkSession

# Initialisation du logger pour suivre le démarrage de Spark
logger = logging.getLogger(__name__)


def create_spark(app_name: str, cfg: Dict[str, Any]) -> SparkSession:
    """Create and configure a SparkSession based on project settings.

    Args:
        app_name: The name of the application (visible in the Spark UI).
        cfg: Dictionary containing the global configuration. It is expected
             to have a 'spark' key with settings like 'master' and 'log_level'.

    Returns:
        A fully configured pyspark.sql.SparkSession instance.
    """
    # Récupération de la section 'spark' (avec fallback vide si absente)
    spark_cfg = cfg.get("spark", {})
    
    master = spark_cfg.get("master", "local[*]")
    logger.info(f"Initializing SparkSession '{app_name}' on master: {master}")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
    )

    # Optimisation pour l'exécution locale :
    # Par défaut Spark utilise 200 partitions pour les shuffles (trop pour ton PC).
    # On réduit ce nombre (ex: 8) pour accélérer les jointures sur petits volumes.
    partitions = str(spark_cfg.get("shuffle_partitions", 8))
    builder = builder.config("spark.sql.shuffle.partitions", partitions)

    # Création effective de la session
    spark = builder.getOrCreate()

    # Configuration du niveau de log (WARN évite de polluer la console)
    log_level = spark_cfg.get("log_level", "WARN")
    spark.sparkContext.setLogLevel(log_level)
    
    logger.info(f"SparkSession created successfully. Log level: {log_level}")

    return spark