"""Training and evaluation for the Bitcoin price prediction models.

This module encapsulates the logic for splitting data into temporal
training and testing sets, building a simple logistic regression model
using Spark MLlib, evaluating its performance, and persisting the
metrics and model artefacts. It can be extended to support more
complex models (e.g. random forests, gradient boosting) via the
configuration file.
"""

from typing import Dict, Any 
from pathlib import Path
import csv
import os 

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def build_train_test(df_labeled: DataFrame, cfg: Dict[str, any]):
    """Split the labeled dataset into training and testing sets based on time.

    Args:
        df_labeled: DataFrame containing features and labels with ``ts_hour``.
        cfg: Loaded configuration dictionary.

    Returns:
        A tuple ``(train_df, test_df)`` where each is a Spark DataFrame.
    """
    train_end = cfg["train"]["train_end_date"]
    test_start = cfg["train"]["test_start_date"]

    train_df = df_labeled.filter(F.col("ts_hour") <= F.lit(train_end))
    test_df = df_labeled.filter(F.col("ts_hour") >= F.lit(test_start))
    return train_df, test_df


def train_and_evaluate(train_df: DataFrame, test_df: DataFrame, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Train a logistic regression classifier and evaluate its performance.

    Builds a simple ML pipeline composed of a ``VectorAssembler``,
    ``StandardScaler`` and ``LogisticRegression`` model. Predictions are
    scored using AUC ROC and accuracy metrics. Metrics are saved to
    a CSV file, and the model is persisted to disk.
    """
    # --- Features -----------------------------------------------------------
    feat_cfg = cfg["features"]
    use_blockchain = feat_cfg.get("use_blockchain", False)
    price_features = feat_cfg.get("price_features", [])
    blockchain_features = feat_cfg.get("blockchain_features", [])

    # colonnes candidates
    feature_cols = list(price_features)
    if use_blockchain:
        feature_cols += blockchain_features

    # on ne garde que les features qui existent vraiment dans le DataFrame
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    # label (créé dans add_labels, généralement "label" ou "label_up")
    label_col = cfg.get("label_col", "label")

    # --- Séparation features prix / features blockchain ---------------------
    # colonnes blockchain effectivement présentes parmi les features
    bc_cols = [c for c in blockchain_features if c in feature_cols]
    # colonnes "coeur" (prix) sur lesquelles on est strict pour les NA
    core_cols = [c for c in feature_cols if c not in bc_cols]

    # --- Config modèle ------------------------------------------------------
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "logistic_regression")
    params = model_cfg.get("params", {})

    output_cfg = cfg.get("output", {})
    output_dir = output_cfg.get("path", "output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Nettoyage NA -------------------------------------------------------
    # 1) on drop uniquement si NA sur les features prix + label
    subset_drop = core_cols + [label_col]
    train_df = train_df.dropna(subset=subset_drop)
    test_df = test_df.dropna(subset=subset_drop)

    # 2) on remplace les NA des features blockchain (issues de la jointure) par 0
    if bc_cols:
        fill_vals = {c: 0.0 for c in bc_cols}
        train_df = train_df.fillna(fill_vals)
        test_df = test_df.fillna(fill_vals)

    # (optionnel) petit debug :
    # print("[DEBUG] n_train après nettoyage :", train_df.count())
    # print("[DEBUG] n_test après nettoyage :", test_df.count())

    # --- Pipeline ML --------------------------------------------------------
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True,
    )

    if model_type == "logistic_regression":
        clf = LogisticRegression(
            labelCol=label_col,
            featuresCol="features",
            maxIter=params.get("maxIter", 50),
            regParam=params.get("regParam", 0.0),
            elasticNetParam=params.get("elasticNetParam", 0.0),
        )
    elif model_type == "gbt":
        clf = GBTClassifier(
            labelCol=label_col,
            featuresCol="features",
            maxDepth=params.get("maxDepth", 5),
            maxIter=params.get("maxIter", 80),
            stepSize=params.get("stepSize", 0.1),
            seed=params.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline(stages=[assembler, scaler, clf])

    # --- Entraînement -------------------------------------------------------
    model = pipeline.fit(train_df)

    # --- Prédiction & métriques --------------------------------------------
    preds = model.transform(test_df)

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = evaluator_auc.evaluate(preds)

    correct = preds.filter(F.col("prediction") == F.col(label_col)).count()
    total = preds.count()
    acc = correct / total if total > 0 else 0.0

    metrics = {
        "auc_roc": float(auc),
        "accuracy": float(acc),
        "n_train": int(train_df.count()),
        "n_test": int(test_df.count()),
    }

    # --- Sauvegarde modèle & métriques -------------------------------------
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model.write().overwrite().save(os.path.join(model_dir, f"{model_type}_pipeline"))

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    return metrics
