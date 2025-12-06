"""Configuration loader for the Bitcoin prediction project.

This module handles the loading and parsing of the YAML configuration file.
It includes error handling for missing files and invalid YAML syntax, ensuring
the pipeline starts with valid parameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

# Initialisation du logger pour ce module
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML file on disk.

    Returns:
        A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid YAML or is not a dictionary.
    """
    cfg_path = Path(path)

    # 1. Vérification de l'existence du fichier
    if not cfg_path.exists():
        msg = f"Configuration file not found at: {cfg_path.absolute()}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading configuration from: {cfg_path}")

    # 2. Chargement sécurisé avec gestion d'erreurs YAML
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML file {cfg_path}: {e}"
        logger.error(msg)
        raise ValueError(msg)

    # 3. Validation basique (on attend un dictionnaire)
    if not isinstance(config, dict):
        msg = f"Invalid config format. Expected a dictionary, got {type(config)}."
        logger.error(msg)
        raise ValueError(msg)

    return config