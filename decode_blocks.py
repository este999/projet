"""Raw Bitcoin Block Parser.

This script is a standalone utility designed to parse raw Bitcoin Core
block files (.dat format). It decodes binary headers to extract:
- Timestamps
- Transaction counts

Usage:
    python3 decode_blocks.py --input-dir data/blocks --output data/blockchain/parsed.parquet
"""

import argparse
import logging
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Generator

import pandas as pd

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BlockParser")

# Magic bytes Bitcoin (Identifiants de réseau)
MAINNET_MAGIC = b"\xf9\xbe\xb4\xd9"
TESTNET_MAGIC = b"\x0b\x11\x09\x07"
MAGICS = {MAINNET_MAGIC, TESTNET_MAGIC}


def decode_varint(b: bytes, offset: int) -> Tuple[int, int]:
    """Decode a Bitcoin variable-length integer (VarInt).

    Args:
        b: Byte sequence.
        offset: Start position.

    Returns:
        Tuple (value, bytes_consumed).
    """
    first = b[offset]
    if first < 0xFD:
        return first, 1
    elif first == 0xFD:
        return struct.unpack_from("<H", b, offset + 1)[0], 3
    elif first == 0xFE:
        return struct.unpack_from("<I", b, offset + 1)[0], 5
    else:
        return struct.unpack_from("<Q", b, offset + 1)[0], 9


def parse_block_header_and_txcount(block_bytes: bytes) -> Tuple[int, int]:
    """Extract timestamp and transaction count from a raw block.

    Structure:
    - Header (80 bytes)
    - Tx Count (VarInt)
    """
    if len(block_bytes) < 81:
        raise ValueError("Block too short")

    header = block_bytes[:80]
    # Timestamp is a uint32 at offset 68 in the header (Little-Endian)
    timestamp = struct.unpack_from("<I", header, 68)[0]

    # Transaction count is a VarInt immediately following the header
    tx_count, _ = decode_varint(block_bytes, 80)
    return timestamp, tx_count


def iter_blocks_in_file(path: Path) -> Generator[Tuple[int, int], None, None]:
    """Yield (timestamp, tx_count) for every block in a .dat file."""
    with path.open("rb") as f:
        while True:
            # Read Magic (4) + Size (4)
            header8 = f.read(8)
            if not header8 or len(header8) < 8:
                break

            magic = header8[:4]
            size = struct.unpack("<I", header8[4:])[0]

            if magic not in MAGICS:
                # End of valid data or corrupted file
                break

            block_bytes = f.read(size)
            if len(block_bytes) < size:
                break

            try:
                ts, tx_count = parse_block_header_and_txcount(block_bytes)
                yield ts, tx_count
            except Exception:
                continue


def build_hourly_stats(input_dir: Path) -> Tuple[List[Dict], int]:
    """Aggregate raw block data into hourly statistics."""
    hourly: Dict[str, Dict] = {}
    n_blocks = 0

    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if not name.endswith(".dat"):
                continue
            
            file_path = Path(root) / name
            logger.info(f"Parsing file: {file_path}")
            
            for ts, tx_count in iter_blocks_in_file(file_path):
                n_blocks += 1

                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                # Truncate to hour
                dt_hour = dt.replace(minute=0, second=0, microsecond=0)
                ts_str = dt_hour.strftime("%Y-%m-%d %H:%M:%S")

                if ts_str not in hourly:
                    hourly[ts_str] = {
                        "timestamp": ts_str,
                        "block_count": 0,
                        "tx_count_hour": 0,
                        "total_value_hour": 0.0, # Placeholder (besoin de parser les Tx complètes pour ça)
                    }

                hourly[ts_str]["block_count"] += 1
                if tx_count is not None:
                    hourly[ts_str]["tx_count_hour"] += int(tx_count)

    # Sort by date
    stats = sorted(hourly.values(), key=lambda x: x["timestamp"])
    return stats, n_blocks


def save_hourly_stats(stats: List[Dict], output_path: str) -> None:
    """Save aggregated stats to Parquet format."""
    cols = ["timestamp", "block_count", "tx_count_hour", "total_value_hour"]

    if not stats:
        logger.warning("No blocks decoded. Creating empty DataFrame.")
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(stats)

    df = df[cols]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(out, index=False)
    logger.info(f"Saved {len(df)} hourly rows to {out}")


def main():
    parser = argparse.ArgumentParser(description="Decode Bitcoin blk*.dat into hourly stats")
    parser.add_argument("--input-dir", required=True, help="Folder containing blk*.dat files")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    stats, n_blocks = build_hourly_stats(input_dir)
    logger.info(f"Total: Decoded {n_blocks} blocks into {len(stats)} hourly records.")
    save_hourly_stats(stats, args.output)


if __name__ == "__main__":
    main()