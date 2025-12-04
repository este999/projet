# BTC Price Direction Prediction

This repository contains an end‑to‑end PySpark pipeline to predict the
short‑term direction of Bitcoin’s price using market data and (optionally)
blockchain‑derived features. It is designed to satisfy the final project
requirements for the ESIEE Big Data Analytics course.

## Structure

```
project-final/
├── data/                 # Input datasets
│   ├── prices/           # Kaggle CSVs with 1‑minute candles
│   ├── blocks/           # Raw blk*.dat files or a decoded Parquet
│   └── processed/        # Intermediate Parquet outputs
├── src/                  # PySpark source code
│   ├── config.py         # YAML config loader
│   ├── spark_utils.py    # Spark session builder
│   ├── ingestion.py      # Functions to load price & blockchain data
│   ├── features.py       # Feature engineering routines
│   ├── modeling.py       # Model training & evaluation
│   └── main.py           # Pipeline orchestrator
├── output/               # Generated metrics, models and logs
├── evidence/             # Plan explanations and Spark UI screenshots
├── bda_project_config.yml # Central configuration file
├── BDA_Project_Report.md  # Project report template
└── run_local.sh           # Convenience script to run locally
```

## Running the pipeline

To run the pipeline on your machine, ensure you have installed the
necessary dependencies (Spark and Python) and that the Kaggle price CSVs
are available under `data/prices/`. Then execute:

```bash
cd project-final
./run_local.sh
```

You can adjust any aspect of the run via the YAML file `bda_project_config.yml`.

## Extending to blockchain data

The baseline pipeline uses only price‑derived features. To incorporate
blockchain metrics, decode the raw blk*.dat files (or download a
precomputed Parquet) into a table with hourly aggregates (e.g. number
of transactions, total transferred value) and set `blockchain_path` in
the YAML config. Then set `use_blockchain: true` and list the
appropriate column names under `blockchain_features`.