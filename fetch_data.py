"""
Data Download Utility: Yahoo Finance Proxy.

This script acts as a fallback data provider when a full Bitcoin node
is not available. It downloads historical market data from Yahoo Finance
and generates proxy metrics for blockchain activity:
- 'Volume' is used as a proxy for 'total_value_hour' and 'tx_count_hour'.
- 'Block count' is simulated as a constant average (6 blocks/hour).

This ensures the pipeline has valid input data covering the required
training period (e.g., 2018-2022).
"""

import logging
import os
import sys

import pandas as pd
import yfinance as yf

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FetchData")

# Configuration des chemins
OUTPUT_DIR = "data/blockchain"
OUTPUT_FILE = "btc_blockchain_hourly_shifted.parquet"
FULL_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)


def fetch_and_process_data():
    """Download daily data and resample to hourly for Spark compatibility."""
    
    # 1. Téléchargement (Daily pour avoir l'historique long)
    start_date = "2018-01-01"
    end_date = "2023-01-01"
    
    logger.info(f"Downloading BTC-USD data from Yahoo Finance ({start_date} to {end_date})...")
    
    try:
        btc = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d", progress=False)
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        sys.exit(1)

    if btc.empty:
        logger.error("Downloaded data is empty. Check your internet connection or date range.")
        sys.exit(1)

    # 2. Nettoyage et Formatage
    # Gestion du MultiIndex souvent renvoyé par les versions récentes de yfinance
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    # Assurance que l'index est temporel
    btc.index = pd.to_datetime(btc.index)

    logger.info("Resampling daily data to hourly (Forward Fill)...")
    # On étale la donnée journalière sur 24h (Approximation valide pour des tendances de fond)
    btc_hourly = btc.resample('1h').ffill()
    btc_hourly = btc_hourly.reset_index()

    # 3. Construction du DataFrame final pour Spark
    df = pd.DataFrame()

    # Standardisation de la colonne date
    # yfinance peut nommer l'index 'Date' ou 'Datetime'
    col_name = 'Date' if 'Date' in btc_hourly.columns else 'Datetime'
    # Fallback si l'index est une colonne sans nom après reset_index
    if col_name not in btc_hourly.columns:
        df['timestamp'] = btc_hourly.iloc[:, 0].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        df['timestamp'] = btc_hourly[col_name].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 4. Création des Proxies (Simulation de données On-Chain)
    # Le Volume d'échange est fortement corrélé à l'activité réseau
    # On divise par 24 pour répartir le volume journalier sur les heures
    volume_hourly = btc_hourly['Volume'] / 24.0
    
    df['total_value_hour'] = volume_hourly
    df['tx_count_hour'] = volume_hourly / 10000.0  # Facteur d'échelle arbitraire pour simuler un "nombre"
    df['block_count'] = 6  # Moyenne théorique du protocole Bitcoin (1 bloc / 10 min)

    # 5. Sauvegarde
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Saving {len(df)} rows to {FULL_PATH}...")
    df.to_parquet(FULL_PATH, index=False)
    logger.info("✅ Success! Data is ready for the Spark pipeline.")


if __name__ == "__main__":
    fetch_and_process_data()