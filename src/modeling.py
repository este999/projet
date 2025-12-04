"""Training and evaluation for the Bitcoin price prediction models.

This module encapsulates the logic for splitting data into temporal
training and testing sets, building a simple logistic regression model
using Spark MLlib, evaluating its performance, and persisting the
metrics and model artefacts. It can be extended to support more
complex models (e.g. random forests, gradient boosting) via the
configuration file.
"""

from typing import Dict
from pathlib import Path
import csv

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
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


def train_and_evaluate(train_df: DataFrame, test_df: DataFrame, cfg: Dict[str, any]) -> Dict[str, float]:
    """Train a logistic regression classifier and evaluate its performance.

    Builds a simple ML pipeline composed of a ``VectorAssembler``,
    ``StandardScaler`` and ``LogisticRegression`` model. Predictions are
    scored using AUC ROC and accuracy metrics. Metrics are saved to
    a CSV file, and the model is persisted to disk.

    Args:
        train_df: DataFrame used for training.
        test_df: DataFrame used for testing.
        cfg: Loaded configuration dictionary.

    Returns:
        A dictionary of metric names and values.
    """
    feat_cfg = cfg["features"]
    input_cols = feat_cfg.get("price_features", []) + (
        feat_cfg.get("blockchain_features", []) if feat_cfg.get("use_blockchain") else []
    )

    assembler = VectorAssembler(inputCols=input_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    lr_params = cfg["model"].get("params", {})
    lr = LogisticRegression(featuresCol="features", labelCol="label_up", **lr_params)

    pipeline = Pipeline(stages=[assembler, scaler, lr])
    model = pipeline.fit(train_df)

    preds = model.transform(test_df)

    # Evaluate AUC ROC
    bc_eval = BinaryClassificationEvaluator(
        labelCol="label_up", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc = bc_eval.evaluate(preds)

    # Evaluate accuracy
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label_up", predictionCol="prediction", metricName="accuracy"
    )
    acc = acc_eval.evaluate(preds)

    metrics = {
        "auc_roc": float(auc),
        "accuracy": float(acc),
        "n_train": train_df.count(),
        "n_test": test_df.count(),
    }

    # Persist metrics to CSV
    metrics_path = Path(cfg["output"]["metrics_csv"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    # Save the trained model
    models_dir = Path(cfg["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(models_dir / "lr_price_baseline"))

    return metrics