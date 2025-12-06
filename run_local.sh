#!/usr/bin/env bash
# =============================================================================
# RUNNER LOCAL - PROJET BDA BITCOIN
# =============================================================================
# Ce script lance le pipeline complet via Spark en mode local.
# Il force l'utilisation de tous les cœurs CPU disponibles (local[*]).
# =============================================================================

# Arrête le script dès qu'une erreur survient
set -euo pipefail

# Configuration
CONFIG_FILE="bda_project_config.yml"
ENTRY_POINT="src/main.py"

echo "-----------------------------------------------------------------------"
echo "🚀 Lancement du Pipeline de Prédiction Bitcoin (BDA Project)"
echo "-----------------------------------------------------------------------"
echo "📂 Config : ${CONFIG_FILE}"
echo "🐍 Script : ${ENTRY_POINT}"
echo "-----------------------------------------------------------------------"

# Vérification de l'existence de Spark
if ! command -v spark-submit &> /dev/null; then
    echo "❌ Erreur : 'spark-submit' est introuvable."
    echo "   Vérifie que SPARK_HOME est bien défini et ajouté au PATH."
    exit 1
fi

# Exécution
# Note : on force 'spark.sql.shuffle.partitions=8' pour optimiser la vitesse
# sur un PC portable (évite de créer 200 partitions vides par défaut).
spark-submit \
  --master "local[*]" \
  --conf spark.sql.shuffle.partitions=8 \
  --conf spark.driver.memory=4g \
  "${ENTRY_POINT}" \
  --config "${CONFIG_FILE}"

echo "-----------------------------------------------------------------------"
echo "✅ Pipeline terminé."
echo "-----------------------------------------------------------------------"