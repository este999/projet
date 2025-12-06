# Environment Specification

**Project:** Bitcoin Price Direction Prediction
**Authors:** Esteban Nabonne & Hadrien Dejonghe

## System Configuration
* **OS:** Linux (WSL2 on Windows) / Ubuntu 22.04
* **Kernel:** Linux 5.15+ (WSL)
* **CPU:** Multi-core (Spark configured for `local[*]`)
* **RAM:** Spark Driver Memory = 4GB

## Software Versions
* **Python:** 3.10+
* **Java:** OpenJDK 21 (Required for Spark 4.0+)
* **Apache Spark:** 4.0.1
* **Hadoop:** Built-in with Spark

## Python Dependencies
Listed in `requirements.txt` (or installed manually):
* `pyspark` (4.0.1)
* `pandas` (2.x)
* `yfinance` (0.2.x) - Data Proxy Source
* `pyarrow` (Active Parquet engine)

## Key Spark Configurations
Defined in `bda_project_config.yml` and `spark_utils.py`:
* `spark.master`: `local[*]`
* `spark.sql.shuffle.partitions`: `8` (Optimized for small local datasets)
* `spark.driver.memory`: `4g`