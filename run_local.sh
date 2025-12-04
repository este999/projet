#!/usr/bin/env bash
# Simple wrapper to execute the Bitcoin prediction pipeline locally.
#
# This script configures Spark to run on all local cores and
# forwards the YAML configuration file path to the Python entry point.

set -euo pipefail

# Path to configuration file. Adjust if necessary.
CONFIG="bda_project_config.yml"

spark-submit \
  --master local[*] \
  --conf spark.sql.shuffle.partitions=8 \
  src/main.py \
  --config "${CONFIG}"