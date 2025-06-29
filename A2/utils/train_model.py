import argparse
import os
import pickle
import warnings
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from dateutil.relativedelta import relativedelta

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Optional libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Project utilities
import sys
import os
sys.path.append('/opt/airflow')
from utils.bronze_processing import ingest_bronze_tables
from utils.silver_processing import (
    clean_financials_table,
    clean_attributes_table,
    clean_clickstream_table,
    clean_loans_table
)
from utils.gold_processing import build_label_store, build_feature_store

warnings.filterwarnings('ignore')


def create_spark_session():
    spark = SparkSession.builder.appName("LoanFeaturePipeline").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def init_datamart(paths=None):
    paths = paths or ["/opt/airflow/datamart/bronze", "/opt/airflow/datamart/silver", "/opt/airflow/datamart/gold"]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"Ensured directory exists: {path}")


def run_data_pipeline(spark):
    ingest_bronze_tables(spark)
    clean_financials_table(spark)
    clean_attributes_table(spark)
    clean_clickstream_table(spark)
    clean_loans_table(spark)
    build_label_store(spark, dpd_cutoff=30, mob_cutoff=3)
    build_feature_store(spark, dpd_cutoff=30, mob_cutoff=3)
    print("Data pipeline completed successfully.")


def load_training_data(spark, snapshot_date):
    fs_path = "/opt/airflow/datamart/gold/feature_store"
    ls_path = "/opt/airflow/datamart/gold/label_store"
    if not os.path.isdir(fs_path) or not os.path.isdir(ls_path):
        raise FileNotFoundError("Gold tables missing. Run data pipeline first.")

    features = spark.read.parquet(fs_path)
    labels = spark.read.parquet(ls_path)
    
    print(f"Total features: {features.count()}")
    print(f"Total labels: {labels.count()}")
    
    # Filter features by snapshot date (use exact match for now)
    features_filtered = features.filter(col("feature_snapshot_date") == snapshot_date)
    print(f"Features for {snapshot_date}: {features_filtered.count()}")

    # Join features and labels and convert to pandas for training
    df = features_filtered.join(labels, on="Customer_ID").toPandas()
    print(f"Training data: {len(df)} records loaded.")
    return df


def prepare_features(df):
    exclude = {"Customer_ID", "feature_snapshot_date", "label", "target"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ["int64", "float64"]]
    X = df[feature_cols].fillna(0)
    y_col = next((c for c in ("target", "label") if c in df.columns), None)
    if not y_col:
        raise ValueError("Target column missing in training data")
    return X, df[y_col], feature_cols


def train_xgboost(X_train, X_val, y_train, y_val, X_full, y_full, oot_periods):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 88,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "gamma": 0.5,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0
    }
    model = xgb.train(params, dtrain, num_boost_round=2000,
                      evals=[(dtrain, "train"), (dval, "val")],
                      early_stopping_rounds=50, verbose_eval=False)
    best_iter = model.best_iteration
    preds = lambda X: model.predict(xgb.DMatrix(X), iteration_range=(0, best_iter))

    metrics = {
        "train_auc": roc_auc_score(y_full, preds(X_full)),
        "oot_avg": sum(roc_auc_score(yo, preds(Xo)) for Xo, yo in oot_periods) / len(oot_periods)
    }
    return model, metrics


def train_lightgbm(X_train, X_val, y_train, y_val, X_full, y_full, oot_periods):
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    params = {"objective":"binary","metric":"auc","verbosity":-1, "seed":88}
    model = lgb.train(params, dtrain, num_boost_round=2000,
                      valid_sets=[dtrain, dval],
                      early_stopping_rounds=50, verbose_eval=False)
    preds = lambda X: model.predict(X)
    metrics = {
        "train_auc": roc_auc_score(y_full, preds(X_full)),
        "oot_avg": sum(roc_auc_score(yo, preds(Xo)) for Xo, yo in oot_periods) / len(oot_periods)
    }
    return model, metrics


def train_catboost(X_train, X_val, y_train, y_val, X_full, y_full, oot_periods):
    model = cb.CatBoostClassifier(iterations=2000, early_stopping_rounds=50,
                                  verbose=False, random_seed=88)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds = lambda X: model.predict_proba(X)[:,1]
    metrics = {
        "train_auc": roc_auc_score(y_full, preds(X_full)),
        "oot_avg": sum(roc_auc_score(yo, preds(Xo)) for Xo, yo in oot_periods) / len(oot_periods)
    }
    return model, metrics


def train_and_select(df, config_name, models):
    X, y, features = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # further split for validation
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42)

    # prepare OOT
    oot_size = len(X_te) // 3
    oot_periods = [(X_te.iloc[i*oot_size:(i+1)*oot_size], y_te.iloc[i*oot_size:(i+1)*oot_size]) for i in range(3)]

    results = {}
    for m in models:
        if m == 'xgboost':
            results[m] = train_xgboost(X_train, X_val, y_train, y_val, X_tr, y_tr, oot_periods)
        elif m == 'lightgbm' and LIGHTGBM_AVAILABLE:
            results[m] = train_lightgbm(X_train, X_val, y_train, y_val, X_tr, y_tr, oot_periods)
        elif m == 'catboost' and CATBOOST_AVAILABLE:
            results[m] = train_catboost(X_train, X_val, y_train, y_val, X_tr, y_tr, oot_periods)
        else:
            print(f"Skipping {m} (not available or unsupported)")

    # choose best by oot_avg
    best_model, best_metrics = max(results.values(), key=lambda tup: tup[1]['oot_avg'])
    return best_model, best_metrics


def save_model(artifact, config_name, target_dir="/opt/airflow/model_bank"):
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{config_name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"Model saved to {path}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--config_name", required=True)
    p.add_argument("--mode", choices=["data_pipeline","train_evaluate","full"], default="full")
    p.add_argument("--train_models", default="xgboost,lightgbm,catboost")
    args = p.parse_args()

    spark = create_spark_session()
    init_datamart()

    if args.mode in ("data_pipeline","full"):
        run_data_pipeline(spark)

    if args.mode in ("train_evaluate","full"):
        df = load_training_data(spark, args.snapshotdate)
        models = [m.strip() for m in args.train_models.split(',')]
        best_model, metrics = train_and_select(df, args.config_name, models)
        artifact = {"model": best_model, "metrics": metrics, "config": args.config_name}
        save_model(artifact, args.config_name)

    spark.stop()


if __name__ == "__main__":
    main()