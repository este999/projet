# Bitcoin Price Direction Prediction (BDA Project)

**Authors:** Esteban Nabonne & Hadrien Dejonghe  
**Course:** Big Data Analytics - ESIEE 2025  

An end-to-end PySpark pipeline to predict Bitcoin price movements using market data and blockchain activity proxies.

---

## 🚀 Quick Start (Reproducibility)

This project includes a **One-Shot Runner** for full reproducibility.

1.  **Install Dependencies:**
    ```bash
    pip install pyspark pandas yfinance pyarrow pyyaml
    ```

2.  **Fetch Data:**
    Downloads market data and generates the blockchain proxy dataset.
    ```bash
    python3 fetch_data.py
    ```

3.  **Run Pipeline:**
    Executes Ingestion, ETL, Training, and Evaluation.
    ```bash
    ./run_local.sh
    ```

---

## 📂 Data Documentation

### 1. Market Prices (Primary Source)
* **Path:** `data/prices/btc_1h_data_2018_to_2025.csv`
* **Description:** 1-minute OHLCV candles aggregated to 1-hour.
* **Source:** Kaggle / Public Crypto Datasets.

### 2. Blockchain Data (Proxy Strategy)
* **Path:** `data/blockchain/btc_blockchain_hourly_shifted.parquet`
* **Method:** Generated via `fetch_data.py` using Yahoo Finance API.
* **Rationale:** Proxies on-chain activity using Trading Volume to bypass the need for a 500GB Full Node synchronization.

### 3. Raw Blocks (Proof of Concept)
* **Archive:** `data/btc_blocks_pruned_1GiB.tar.gz` (Not included in Git due to size).
* **Live Folder:** `data/blocks/blocks/`
* **Parser:** `src/decode_blocks.py` is provided to demonstrate capability to parse raw `.dat` Bitcoin Core files.
* **Usage:** Run `python3 decode_blocks.py --input-dir data/blocks/blocks --output data/blockchain/debug.parquet` to verify.

---

## 📊 Outputs & Evidence

* **Metrics:** `output/metrics.csv` contains the final Accuracy and AUC scores.
* **Logs:** `output/logs/` (if enabled) or Console Output.
* **Spark Evidence:** See `evidence/` folder for:
    * `spark_dag.png`: The SQL execution plan graph.
    * `spark_jobs.png`: Timeline of Spark stages.
    * `explain_plan.txt`: Textual representation of the physical plan (`explain("formatted")`).

---

## ⚖️ Licenses & Citations

* **Code:** MIT License.
* **Data (Yahoo Finance):** Used via `yfinance` library for educational purposes.
* **Data (Bitcoin Core):** Raw block format follows the Bitcoin Protocol specification (MIT).
* **Spark:** Apache License 2.0.