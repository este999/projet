"""Model training and evaluation module.

This module encapsulates the machine learning workflow:
1. Splitting data into time-based Train/Test sets.
2. Building a Spark ML Pipeline (Assembler -> Scaler -> Model).
3. Training the model (Random Forest, GBT, etc.).
4. Evaluating performance (Accuracy, AUC).
5. Saving metrics and model artifacts.
"""

import csv
import logging
import os
from typing import Any, Dict, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression, 
    GBTClassifier, 
    RandomForestClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, 
    MulticlassClassificationEvaluator
)

# Initialisation du logger pour suivre l'entraînement
logger = logging.getLogger(__name__)


def build_train_test(df_labeled: DataFrame, cfg: Dict[str, Any]) -> Tuple[DataFrame, DataFrame]:
    """Split the dataset into training and testing sets based on configuration dates.

    Args:
        df_labeled: DataFrame containing features and the label column.
        cfg: Configuration dictionary with 'train' parameters.

    Returns:
        Tuple (train_df, test_df).
    """
    train_cfg = cfg.get("train", {})
    train_end = train_cfg.get("train_end_date")
    test_start = train_cfg.get("test_start_date")

    logger.info(f"Splitting data... Train up to {train_end}, Test from {test_start}")

    # Découpage temporel strict basé sur la colonne 'ts_hour'
    train_df = df_labeled.filter(F.col("ts_hour") <= F.lit(train_end))
    test_df = df_labeled.filter(F.col("ts_hour") >= F.lit(test_start))

    logger.info(f"Train set size: {train_df.count()} rows")
    logger.info(f"Test set size: {test_df.count()} rows")

    return train_df, test_df


def train_and_evaluate(train_df: DataFrame, test_df: DataFrame, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Train the model defined in config and evaluate on test set.

    Args:
        train_df: Training DataFrame.
        test_df: Testing DataFrame.
        cfg: Configuration dictionary.

    Returns:
        Dictionary of metrics (auc_roc, accuracy, n_train, n_test).
    """
    # --- 1. Préparation des Features ----------------------------------------
    feat_cfg = cfg.get("features", {})
    use_blockchain = feat_cfg.get("use_blockchain", False)
    
    # Récupération de la liste des features de prix
    feature_cols = list(feat_cfg.get("price_features", []))
    
    # Ajout conditionnel des features blockchain
    blockchain_cols = feat_cfg.get("blockchain_features", [])
    if use_blockchain:
        feature_cols += blockchain_cols
        logger.info(f"Using blockchain features: {blockchain_cols}")

    # Vérification de sécurité : on ne garde que les colonnes qui existent vraiment
    existing_cols = train_df.columns
    valid_features = [c for c in feature_cols if c in existing_cols]
    
    # Alerte si des features demandées sont introuvables
    if len(valid_features) < len(feature_cols):
        missing = set(feature_cols) - set(valid_features)
        logger.warning(f"Some features are missing from dataframe and ignored: {missing}")

    label_col = cfg.get("label_col", "label_up")

    # --- 2. Nettoyage des Données (NA handling) -----------------------------
    # Stratégie :
    # - Features Blockchain manquantes -> Remplacées par 0.0 (hypothèse d'absence d'activité ou de données)
    # - Features Prix manquantes -> Suppression de la ligne (donnée corrompue)
    
    if use_blockchain:
        # On ne remplit que les colonnes blockchain présentes
        fill_dict = {c: 0.0 for c in blockchain_cols if c in valid_features}
        if fill_dict:
            train_df = train_df.fillna(fill_dict)
            test_df = test_df.fillna(fill_dict)

    # Suppression des lignes incomplètes restantes (prix ou label manquants)
    train_df = train_df.dropna(subset=valid_features + [label_col])
    test_df = test_df.dropna(subset=valid_features + [label_col])

    # --- 3. Construction du Pipeline ML -------------------------------------
    # Assemblage des features en un vecteur unique
    assembler = VectorAssembler(
        inputCols=valid_features,
        outputCol="features_raw",
        handleInvalid="skip"  # Sécurité : ignore les lignes malformées silencieusement
    )

    # Standardisation (Moyenne=0, Variance=1) -> Aide beaucoup les modèles linéaires
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True,
    )

    # Instanciation du modèle selon la config
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "random_forest")
    params = model_cfg.get("params", {})

    logger.info(f"Initializing model: {model_type} with params: {params}")

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
            maxIter=params.get("maxIter", 50),
            stepSize=params.get("stepSize", 0.1),
            seed=params.get("seed", 42),
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            labelCol=label_col,
            featuresCol="features",
            numTrees=params.get("numTrees", 100),
            maxDepth=params.get("maxDepth", 10),
            seed=params.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown model type in config: {model_type}")

    pipeline = Pipeline(stages=[assembler, scaler, clf])

    # --- 4. Entraînement & Prédiction ---------------------------------------
    logger.info("Training started...")
    model = pipeline.fit(train_df)
    logger.info("Training finished. Evaluating on test set...")
    
    preds = model.transform(test_df)

    # --- 5. Calcul des Métriques --------------------------------------------
    # AUC-ROC (Capacité à distinguer les classes)
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = evaluator_auc.evaluate(preds)

    # Accuracy (Taux de bonnes prédictions)
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy",
    )
    acc = evaluator_acc.evaluate(preds)

    metrics = {
        "auc_roc": float(auc),
        "accuracy": float(acc),
        "n_train": int(train_df.count()),
        "n_test": int(test_df.count()),
    }
    
    logger.info(f"Evaluation Results: {metrics}")

    # --- 6. Sauvegarde des Résultats ----------------------------------------
    output_cfg = cfg.get("output", {})
    
    # 6.1 Sauvegarde des métriques dans un CSV
    metrics_path = output_cfg.get("metrics_csv", "output/metrics.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    try:
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        logger.info(f"Metrics saved to {metrics_path}")
    except IOError as e:
        logger.error(f"Failed to save metrics CSV: {e}")

    # 6.2 Sauvegarde du modèle complet (Pipeline)
    models_dir = output_cfg.get("models_dir", "output/models")
    model_path = os.path.join(models_dir, f"{model_type}_pipeline")
    
    try:
        model.write().overwrite().save(model_path)
        logger.info(f"Model artifact saved to {model_path}")
    except Exception as e:
        logger.warning(f"Could not save Spark model (folder might be locked/existing): {e}")

    return metrics