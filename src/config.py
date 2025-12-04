"""Configuration loader for the Bitcoin prediction project.

This module provides a helper to load a YAML configuration file into a
dictionary. The configuration file is used to parameterise the Spark
pipeline, including data paths, feature definitions, model settings and
output locations.

The configuration is loaded lazily at runtime by the main entry point.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file on disk.

    Returns:
        A dictionary representing the YAML configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)