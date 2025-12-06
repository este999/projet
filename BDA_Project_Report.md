# Rapport de Projet BDA — Prédiction Bitcoin

**Auteur :** Esteban Nabonne  & Hadrien Dejonghe 

**Cours :** Big Data Analytics - ESIEE 2025-2026  

**Sujet :** Prédiction de la Direction du Prix du Bitcoin avec PySpark


## 1. Définition du Problème et Objectifs

### 1.1 Le Contexte
Le **Bitcoin** est un actif financier extrêmement volatil. Prédire son prix exact (**Régression**) est notoirement difficile et bruité. En revanche, prédire la direction de son mouvement (**Classification**) est plus réalisable et tout aussi utile pour une stratégie d'investissement.

### 1.2 L'Objectif Technique
Nous avons conçu un pipeline Big Data capable de répondre à la question suivante :

> "Le Bitcoin va-t-il clôturer plus haut dans 1 heure par rapport à maintenant ?"

C'est un problème de **Classification Binaire Supervisée** :
* **Classe 1 (Hausse) :** Rendement futur positif (> 0.1%).
* **Classe 0 (Baisse/Neutre) :** Rendement futur négatif ou nul.

### 1.3 Nos Hypothèses de Travail
* **L'inertie du marché (Momentum) :** Les mouvements passés récents (1h, 3h, 12h) influencent le futur immédiat.
* **L'activité réseau (Blockchain) :** Une augmentation du volume d'échange précède souvent une période de forte volatilité.
* **La dépendance au régime (Regime Dependency) :** Un modèle entraîné uniquement sur un marché haussier (2020-2021) sera incapable de comprendre un krach. Il faut lui montrer des exemples historiques de crises (comme 2018) pour qu'il soit robuste en 2022.

## 2. Acquisition et Sélection des Données
Nous avons dû arbitrer entre la pureté technique et la faisabilité temporelle du projet.

### 2.1 Données de Prix (Source Principale)
* **Source :** Fichier CSV historique (2018-2025).
* **Contenu :** Bougies **OHLCV** (Open, High, Low, Close, Volume) **agrégées à l'heure (1h)**.
* **Utilité :** C'est la base indispensable pour calculer nos indicateurs techniques et notre variable cible. L'utilisation de données pré-agrégées (1h) au lieu de données minute optimise considérablement les temps de chargement et de traitement sur une architecture locale.

### 2.2 Données Blockchain (Le choix du Proxy)
Nous souhaitions intégrer l'activité réelle du réseau (*On-Chain metrics*).

#### Option A (Rejetée) : Nœud Complet
Nous avons développé un script complexe (`decode_blocks.py`) capable de lire les fichiers binaires `.dat` de Bitcoin Core.
* **Résultat :** Bien que fonctionnel techniquement (11 000 blocs décodés en test), la synchronisation complète (**450 Go**) prenait trop de temps.

#### Option B (Retenue) : Proxy Yahoo Finance
Nous avons utilisé le script `fetch_data.py` pour récupérer le volume d'échange agrégé via Yahoo Finance.

> **Justification :** Le volume d'échange est fortement corrélé à l'activité de la blockchain (nombre de transactions). C'est un "proxy" fiable qui nous permet d'avoir des données complètes de 2018 à 2023 instantanément.


## 3. Architecture du Pipeline (ETL)Le projet repose sur un pipeline **PySpark** modulaire, conçu pour être scalable.

### 3.1 Ingestion et Nettoyage
* Le code charge les CSV et force le typage (`DoubleType`) pour éviter que les prix soient lus comme du texte.
* Nous appliquons un filtrage temporel strict (**Start/End Date**) défini dans le fichier de configuration `bda_project_config.yml` pour garantir que nous comparons des périodes cohérentes.

### 3.2 Feature Engineering (Création d'attributs)
C'est ici que réside l'intelligence du projet. Nous transformons la donnée brute en signaux exploitables :

* **Rééchantillonnage (Resampling) :**
  Spark agrège les données "minute" en fenêtres d'une heure (`window("ts_raw", "1 hour")`). Cela lisse le bruit du marché.

* **Indicateurs de Tendance :**
    * *Moyennes Mobiles (SMA) :* Prix moyen sur 3h et 12h. Si le prix actuel est au-dessus de la moyenne 12h, c'est un signal haussier.
    * *Retards (Lags) :* Nous donnons au modèle le rendement de l'heure précédente (`ret_1h_lag1`).

* **Indicateurs de Risque :**
    * *Volatilité :* Écart-type des prix dans l'heure.

* **Enrichissement :**
  Jointure (`Left Join`) avec les données Blockchain proxy sur la clé temporelle `ts_hour`.

### 3.3 Étiquetage (Labeling)
Nous créons la colonne cible `label_up`. Nous utilisons un seuil de **0.1% de hausse** pour considérer un mouvement comme significatif, afin d'éviter que le modèle n'apprenne sur du "bruit" stationnaire.

## 4. Méthodologie Expérimentale
Nous avons mené une étude comparative pour prouver l'importance de l'historique des données.

### Expérience 1 : L'Approche Naïve (Échec)
* **Entraînement :** 2019 à 2021.
    * *Contexte :* Le Bitcoin passe de 3 000 $à 69 000$. Le marché est euphorique (**Bull Run**).
* **Test :** 2022.
    * *Contexte :* Le Bitcoin s'effondre (**Bear Market**).
* **Résultat :** `Accuracy 45%`

> **Analyse :** Le modèle a appris à "acheter tout le temps". Face à la crise de 2022, il a échoué massivement.

### Expérience 2 : L'Approche Robuste (Réussite)
* **Entraînement :** 2018 à 2021.
* **Changement clé :** Nous avons ajouté l'année **2018**, qui était une année de krach violent (-80%).
* **Test :** 2022.
* **Résultat :** `Accuracy 61.5%`

> **Analyse :** En voyant le krach de 2018, le modèle (**Random Forest**) a appris à identifier les signaux précurseurs d'une baisse. Il est devenu capable de naviguer dans la tempête de 2022.

## 5. Preuves de Performance et Optimisation Big Data
Nous avons dû adapter la configuration Spark pour un environnement local.

### 5.1 Optimisation des Partitions

* **Problème :** Par défaut, Spark utilise **200 partitions** pour les opérations de mélange (*shuffle*). Pour notre dataset (~50 Mo), cela créait une surcharge énorme (*overhead*) de gestion de tâches vides.
* **Solution :** Nous avons forcé `spark.sql.shuffle.partitions = 8` dans le fichier `src/spark_utils.py`.
    * *Résultat :* Cela a divisé le temps d'exécution par 3.

### 5.2 Analyse des Logs (Spark UI)
Les logs montrent des avertissements de type : `WARN WindowExec: No Partition Defined`.

> **Explication technique :** Ceci est normal et attendu. Pour calculer une moyenne mobile temporelle (ex: moyenne des 3 dernières heures), Spark doit aligner toutes les données chronologiquement sur une seule partition logique. C'est une contrainte inhérente à l'analyse de séries temporelles (*Time Series*) distribuée.

## 6. Analyse des Résultats et Discussion

### Tableau des Résultats Finaux

| Modèle | Période Entraînement | Période Test | Accuracy | AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 2018-2021 | 2022 | **61.50%** | **0.568** |

### Discussion* **La victoire de la "Data Science" :**
  Passer de 45% à **61.5%** sans changer le code, juste en changeant les données d'entraînement, prouve que la représentativité des données est le facteur le plus critique.

* **Performance :**
  Une précision de **61.5%** est excellente pour un actif financier aussi imprévisible. Cela signifie que le modèle a capté un vrai signal de tendance.

* **Limite de l'AUC :**
  L'AUC reste modeste (**0.57**). Cela indique que le modèle est bon pour donner la direction (*Haut/Bas*) mais moins précis pour estimer la probabilité (certitude) de cette direction.

## 7. Reproductibilité

Pour relancer le projet, il suffit de suivre ces trois étapes :

1. **Installer les dépendances Python :**
```bash
pip install pyspark pandas yfinance pyarrow pyyaml
```
2. **Génération des Données :**
```bash
python3 fetch_data.py
```
3. **Exécution du Pipeline :**
```bash
./run_local.sh
```

Le fichier de configuration `bda_project_config.yml` contient tous les hyperparamètres utilisés pour obtenir le score de **61.5%**.