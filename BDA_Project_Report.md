# BTC Price Direction Prediction Project

## Introduction

Predicting the short‑term direction of Bitcoin’s price is an interesting problem
for both traders and researchers. This project tackles it through an
end‑to‑end big data pipeline built around Apache Spark. We combine two
complementary datasets – minute‑level candles from Kaggle and a
blockchain‑derived dataset – to explore whether on‑chain activity can help
forecast hourly price movements. The approach follows the final project
brief from ESIEE’s Big Data Analytics course【123†L0-L17】.

## Data acquisition

### Market prices

Historical Bitcoin prices were retrieved from Kaggle, specifically a
dataset of one‑minute OHLCV candles. These CSV files are stored under
`data/prices/` and are ingested by Spark using the schema configured in
`bda_project_config.yml`. The ingestion converts timestamps to proper
`timestamp` types and filters rows to the analysis period (2019–2022).

### Blockchain data

The project brief mandates using raw blockchain blocks (`blk*.dat`) or a
faithful derivative【122†L0-L15】. A guide in the dataset package explains how to
run Bitcoin Core in prune mode to archive about 1 GiB of raw blocks【120†filecite】. For
this report, we concentrate on the price‑only baseline; the blockchain
ingestion is implemented as a stub that can load Parquet files once a
decoded dataset becomes available.

## Pipeline overview

The pipeline is orchestrated by `src/main.py` and parameterised via
`bda_project_config.yml`. The major steps are:

1. **Ingestion:** Load price candles from CSV and (optionally) blockchain
   aggregates from Parquet.
2. **Feature engineering:** Resample minute candles into hourly bars,
   computing aggregate statistics and derived features (returns,
   volatility, volume, high/low ratios). Blockchain features are joined
   if configured.
3. **Labelling:** For each hourly bar, compute the one‑hour future return
   and label it as `1` when the return is positive, `0` otherwise.
4. **Training/testing split:** Split the labelled dataset chronologically
   according to the dates in the YAML config.
5. **Model training and evaluation:** Use Spark ML to assemble features,
   scale them and fit a logistic regression classifier. Evaluate using
   AUC ROC and accuracy on the test set. Metrics and the model artefact
   are persisted to `output/`.

## Implementation details

### Configuration

The pipeline reads all settings from a YAML file. This makes it easy to
adjust data paths, feature sets, model hyperparameters and date ranges
without changing the code. The default configuration provided with this
project uses only price features and disables blockchain features.

### Feature engineering

Minute candles are aggregated to hourly windows using Spark’s
`window` function. The first `open`, last `close`, maximum `high`,
minimum `low`, total `volume` and population standard deviation of the
`close` price are computed for each hour. Derived features include the
log return over the previous hour and the ratio of `high` to `low`.

### Modelling

For the baseline we choose logistic regression for its interpretability
and simplicity. A `VectorAssembler` and `StandardScaler` prepare the
feature vectors. Accuracy and AUC scores are written to a CSV file and
the trained model is saved for reproducibility.

## Results

After running `run_local.sh`, the pipeline produces metrics such as AUC
ROC and accuracy in `output/metrics.csv`. With price‑only features on
2019–2022 data, the baseline model achieved an AUC of … (replace with
your results). Adding blockchain features should allow exploration of
whether on‑chain activity provides incremental predictive power.

## Conclusion and future work

This project delivers a full reproducible workflow for short‑term
Bitcoin price direction prediction. It demonstrates how to ingest and
aggregate large time‑series datasets with Spark, engineer meaningful
features, train ML models and report results. Future work will include
parsing raw block files to extract transaction graphs, engineering
additional on‑chain indicators (e.g. UTXO age distributions, whale
alerts)【122†L0-L27】, and experimenting with more sophisticated models such
as gradient boosted trees or deep learning.