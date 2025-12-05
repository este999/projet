#!/usr/bin/env python3
"""
decode_blocks.py
=================

This helper script is intended to **decode raw Bitcoin block files**
(``blkNNNNN.dat``) produced by Bitcoin Core (e.g. via prune mode) and
produce an aggregated, hourly view of on‑chain activity.  The goal is to
transform the binary blockchain data into a simple Parquet dataset
containing per‑hour metrics that can be joined with price features in
Spark for the final project.

**What it does**

* Walks a directory containing raw block files (the ``blk*.dat`` files).
* Reads each block by scanning for the network magic header and size.
* Extracts the block timestamp and counts the number of transactions.
* Sums the total value of all transaction outputs in the block.
* Buckets the blocks by the hour of their timestamp (UTC) and
  aggregates the transaction count and total output value across all
  blocks in that hour.
* Writes the resulting per‑hour metrics to a Parquet file (or CSV if
  specified).

This script does **not** implement a full Bitcoin protocol parser—it
extracts just enough information for the project: the block timestamp,
transaction count and total output value.  It should work for blocks up
to version 4 and does not attempt to compute transaction fees.

Usage
-----

Run this script from the repository root as follows::

    python decode_blocks.py --input-dir data/btc_blocks_pruned_1GiB/blocks \
                            --output data/blockchain/blockchain_features_hourly.parquet

This will create the output directory ``data/blockchain`` if it does
not exist and produce a Parquet file with columns:

* ``ts_hour``: The hour (UTC) of the blocks, truncated to the hour.
* ``tx_count_hour``: Total number of transactions across all blocks in
  that hour.
* ``total_output_hour``: Total value of transaction outputs in Bitcoin
  (floating, not satoshis) across all blocks in that hour.

You can then update your ``bda_project_config.yml`` with::

    data:
      blockchain_path: "data/blockchain/blockchain_features_hourly.parquet"
    features:
      use_blockchain: true
      blockchain_features:
        - "tx_count_hour"
        - "total_output_hour"

and the pipeline will join these features with your price data.

Notes
-----

* The script is deliberately single‑threaded and streams through the
  block files—it will take some time to run on ~1 GiB of blocks but
  will not consume excessive memory.
* The total output value is computed per block by summing the values of
  all transaction outputs.  Each output value is an 8‑byte little
  endian integer representing satoshis (10⁻⁸ BTC).  The script
  converts this to BTC by dividing by 1e8.
* If your raw block files are for testnet, you may need to adjust the
  network magic constant (see ``NETWORK_MAGIC`` below).

"""

import argparse
import os
import struct
import datetime
from collections import defaultdict
from typing import Iterator, Tuple, Dict

import pandas as pd


# Bitcoin network magic for mainnet (little endian order when reading
# from file).  If you have testnet data, change this to
# b'\x0b\x11\x09\x07'.
NETWORK_MAGIC = b"\xf9\xbe\xb4\xd9"


def read_varint(buf: memoryview, offset: int) -> Tuple[int, int]:
    """Parse a Bitcoin variable length integer from a buffer.

    Args:
        buf: A memoryview over the bytes we are reading.
        offset: Current offset into the buffer.

    Returns:
        A tuple (value, new_offset) where ``value`` is the decoded
        integer and ``new_offset`` is the offset after reading the
        varint.
    """
    first = buf[offset]
    offset += 1
    if first < 0xFD:
        return first, offset
    if first == 0xFD:
        # uint16
        value = struct.unpack_from('<H', buf, offset)[0]
        offset += 2
        return value, offset
    if first == 0xFE:
        # uint32
        value = struct.unpack_from('<I', buf, offset)[0]
        offset += 4
        return value, offset
    # first == 0xFF: uint64
    value = struct.unpack_from('<Q', buf, offset)[0]
    offset += 8
    return value, offset


def parse_block(block_data: bytes) -> Tuple[int, int, float]:
    """Parse a single block and extract simple metrics.

    Args:
        block_data: Raw block bytes (not including magic/size).

    Returns:
        A tuple (timestamp, tx_count, total_output_btc).
    """
    # Block header is 80 bytes: version(4) + prev hash(32) + merkle(32)
    # + timestamp(4) + bits(4) + nonce(4)
    if len(block_data) < 80:
        raise ValueError("Block data too short to contain a header")
    # timestamp is little endian uint32 at offset 68
    timestamp = struct.unpack_from('<I', block_data, 68)[0]

    # Move past the 80 byte header
    offset = 80
    buf = memoryview(block_data)

    # Read transaction count (varint)
    tx_count, offset = read_varint(buf, offset)

    total_output_value_sat = 0  # accumulate satoshis
    # Loop over transactions
    for _ in range(tx_count):
        # Version (4 bytes)
        offset += 4
        # Input count (varint)
        n_inputs, offset = read_varint(buf, offset)
        # Skip each input
        for _ in range(n_inputs):
            # Previous txid (32 bytes) + vout (4 bytes)
            offset += 36
            # Script length (varint)
            script_len, offset = read_varint(buf, offset)
            # Script
            offset += script_len
            # Sequence (4 bytes)
            offset += 4
        # Output count (varint)
        n_outputs, offset = read_varint(buf, offset)
        for _ in range(n_outputs):
            # value (8 bytes little endian)
            value = struct.unpack_from('<q', buf, offset)[0]
            total_output_value_sat += value
            offset += 8
            # script length (varint)
            script_len, offset = read_varint(buf, offset)
            # script
            offset += script_len
        # locktime (4 bytes)
        offset += 4

    # Convert satoshis to BTC
    total_output_btc = total_output_value_sat / 1e8
    return timestamp, tx_count, total_output_btc


def iter_blocks_from_file(f) -> Iterator[bytes]:
    """Yield raw block payloads from an open file handle.

    Args:
        f: Binary file handle positioned at the beginning of a blk*.dat
           file.

    Yields:
        The raw block payload (without magic and size).
    """
    # We read until EOF.  Each block is preceded by magic (4 bytes)
    # followed by size (4 bytes little endian).
    magic_len = len(NETWORK_MAGIC)
    while True:
        magic = f.read(magic_len)
        if not magic or len(magic) < magic_len:
            break
        if magic != NETWORK_MAGIC:
            # The dataset may contain zero‐padding; skip until magic
            continue
        size_bytes = f.read(4)
        if len(size_bytes) < 4:
            break
        block_size = struct.unpack('<I', size_bytes)[0]
        block_data = f.read(block_size)
        if len(block_data) < block_size:
            break
        yield block_data


def process_block_directory(input_dir: str) -> Dict[datetime.datetime, Dict[str, float]]:
    """Scan through all blk*.dat files and aggregate hourly metrics.

    Returns a dictionary keyed by UTC hour (datetime with minutes,
    seconds, microseconds set to zero) mapping to metrics:
        {
            'tx_count_hour': total transaction count,
            'total_output_hour': total output value in BTC,
        }
    """
    hourly_stats: Dict[datetime.datetime, Dict[str, float]] = defaultdict(lambda: {'tx_count_hour': 0, 'total_output_hour': 0.0})
    files = [f for f in os.listdir(input_dir) if f.startswith('blk') and f.endswith('.dat')]
    files.sort()
    for filename in files:
        path = os.path.join(input_dir, filename)
        with open(path, 'rb') as f:
            for block in iter_blocks_from_file(f):
                try:
                    ts, tx_count, total_output_btc = parse_block(block)
                except Exception:
                    # Skip malformed blocks
                    continue
                hour = datetime.datetime.utcfromtimestamp(ts).replace(minute=0, second=0, microsecond=0)
                stat = hourly_stats[hour]
                stat['tx_count_hour'] += tx_count
                stat['total_output_hour'] += total_output_btc
    return hourly_stats


def save_hourly_stats(hourly_stats: Dict[datetime.datetime, Dict[str, float]], output_path: str) -> None:
    """Save the aggregated hourly stats to a Parquet or CSV file.

    The file type is determined by the extension of ``output_path``.
    Supported extensions: ``.parquet``, ``.csv``.
    """
    # Convert dictionary to DataFrame
    data = []
    for hour, metrics in sorted(hourly_stats.items()):
        row = {'ts_hour': hour, 'tx_count_hour': metrics['tx_count_hour'], 'total_output_hour': metrics['total_output_hour']}
        data.append(row)
    df = pd.DataFrame(data)
    ext = os.path.splitext(output_path)[1].lower()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if ext == '.parquet':
        df.to_parquet(output_path, index=False)
    elif ext == '.csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {ext}. Use .parquet or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode raw Bitcoin block files and aggregate hourly metrics")
    parser.add_argument('--input-dir', required=True, help='Directory containing blk*.dat files')
    parser.add_argument('--output', required=True, help='Output file path (parquet or csv)')
    args = parser.parse_args()

    hourly_stats = process_block_directory(args.input_dir)
    save_hourly_stats(hourly_stats, args.output)
    print(f"Wrote aggregated blockchain metrics to {args.output}")


if __name__ == '__main__':
    main()