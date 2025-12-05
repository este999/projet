#!/usr/bin/env python3
import argparse
import os
import struct
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

# Magic bytes Bitcoin mainnet / testnet
MAINNET_MAGIC = b"\xf9\xbe\xb4\xd9"
TESTNET_MAGIC = b"\x0b\x11\x09\x07"
MAGICS = {MAINNET_MAGIC, TESTNET_MAGIC}


def decode_varint(b: bytes, offset: int) -> Tuple[int, int]:
    """Decode Bitcoin varint starting at offset, return (value, bytes_used)."""
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
    """
    Parse a block:
    - header = 80 bytes, timestamp = 4 bytes at offset 68 (little-endian)
    - then a varint = number of transactions
    """
    if len(block_bytes) < 81:
        raise ValueError("Block too short")

    header = block_bytes[:80]
    # timestamp is a uint32 at offset 68 in the header
    timestamp = struct.unpack_from("<I", header, 68)[0]

    # varint with tx count just after the 80-byte header
    tx_count, _ = decode_varint(block_bytes, 80)
    return timestamp, tx_count


def iter_blocks_in_file(path: Path):
    """
    Iterate over (timestamp, tx_count) for each block in a blkNNNNN.dat file.
    Format Bitcoin Core:
    [4 bytes magic][4 bytes size][size bytes block]...
    """
    with path.open("rb") as f:
        while True:
            header8 = f.read(8)
            if not header8:
                break
            if len(header8) < 8:
                break

            magic = header8[:4]
            size = struct.unpack("<I", header8[4:])[0]

            if magic not in MAGICS:
                # Not a standard blk file -> stop
                break

            block_bytes = f.read(size)
            if len(block_bytes) < size:
                # Truncated file -> stop
                break

            try:
                ts, tx_count = parse_block_header_and_txcount(block_bytes)
                yield ts, tx_count
            except Exception:
                # Ignore malformed blocks
                continue


def build_hourly_stats(input_dir: Path):
    """
    Go through all .dat in input_dir and aggregate per UTC hour.

    Returns:
      - list of dicts with:
          timestamp (string "YYYY-MM-DD HH:00:00")
          block_count
          tx_count_hour
          total_value_hour (placeholder = 0.0)
      - total number of decoded blocks
    """
    hourly: Dict[str, Dict] = {}
    n_blocks = 0

    for root, _dirs, files in os.walk(input_dir):
        for name in sorted(files):
            if not name.endswith(".dat"):
                continue
            file_path = Path(root) / name
            print(f"[decode_blocks] Parsing {file_path}")
            for ts, tx_count in iter_blocks_in_file(file_path):
                n_blocks += 1

                # ts is a Unix timestamp (seconds)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                dt_hour = dt.replace(minute=0, second=0, microsecond=0)

                # On enlève la timezone et on garde un string propre
                dt_hour_naive = dt_hour.replace(tzinfo=None)
                ts_str = dt_hour_naive.strftime("%Y-%m-%d %H:%M:%S")

                if ts_str not in hourly:
                    hourly[ts_str] = {
                        "timestamp": ts_str,           # très important: string
                        "block_count": 0,
                        "tx_count_hour": 0,
                        "total_value_hour": 0.0,       # placeholder
                    }

                hourly[ts_str]["block_count"] += 1
                if tx_count is not None:
                    hourly[ts_str]["tx_count_hour"] += int(tx_count)

    stats = sorted(hourly.values(), key=lambda x: x["timestamp"])
    return stats, n_blocks


def save_hourly_stats(stats: List[Dict], output_path: str) -> None:
    """Save as parquet with simple types (timestamp = string)."""
    cols = ["timestamp", "block_count", "tx_count_hour", "total_value_hour"]

    if stats:
        df = pd.DataFrame(stats)
    else:
        print("[decode_blocks] WARNING: no data decoded, empty parquet with schema only.")
        df = pd.DataFrame(columns=cols)

    df = df[cols]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[decode_blocks] Saved {len(df)} hourly rows to {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Decode Bitcoin blk*.dat into hourly blockchain stats")
    p.add_argument(
        "--input-dir",
        required=True,
        help="Répertoire contenant les fichiers blk*.dat (ex: data/btc_blocks_pruned_1GIB)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Chemin de sortie parquet (ex: data/blockchain/btc_blockchain_hourly.parquet)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    stats, n_blocks = build_hourly_stats(input_dir)
    print(f"[decode_blocks] Decoded {n_blocks} blocks into {len(stats)} hourly rows.")
    save_hourly_stats(stats, args.output)


if __name__ == "__main__":
    main()
