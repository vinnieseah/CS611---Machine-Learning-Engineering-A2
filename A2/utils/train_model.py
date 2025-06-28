# main.py

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from dateutil.relativedelta import relativedelta

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data processing imports
from utils.bronze_processing import ingest_bronze_tables
from utils.silver_processing import (
    clean_financials_table,
    clean_attributes_table,
    clean_clickstream_table,
    clean_loans_table
)
from utils.gold_processing import (
    build_label_store,
    build_feature_store
)

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Robust imports for optional ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"Warning: LightGBM not available ({e}). LightGBM models will be skipped.")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. CatBoost models will be skipped.")

import warnings
warnings.filterwarnings('ignore')

def create_spark_session():
    spark = SparkSession.builder\
        .appName("LoanFeaturePipeline")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def init_datamart():
    for layer in ["datamart/bronze", "datamart/silver", "datamart/gold", "model_bank"]:
        os.makedirs(layer, exist_ok=True)

def load_training_data(spark, snapshot_date):
    """Load features and labels for model training using the 3-month time offset"""
    print(f"Loading training data for {snapshot_date}")
    
    try:
        # Load the pre-built feature store and label store
        feature_store_path = "datamart/gold/feature_store"
        label_store_path = "datamart/gold/label_store"
        
        if not os.path.exists(feature_store_path):
            raise FileNotFoundError(f"Feature store not found at {feature_store_path}. Please run data pipeline first.")
        
        if not os.path.exists(label_store_path):
            raise FileNotFoundError(f"Label store not found at {label_store_path}. Please run data pipeline first.")
        
        # The gold layer is designed with a 3-month offset between features and labels
        # We need to create a training dataset by joining features with future labels
        
        # Convert snapshot_date to datetime for calculation
        snapshot_dt = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
        
        # Load feature store - limit to features up to snapshot date
        features_sdf = spark.read.parquet(feature_store_path)
        features_sdf = features_sdf.filter(col("feature_snapshot_date") <= snapshot_date)
        print(f"Features available: {features_sdf.count()} records")
        
        # Load label store - load ALL available labels (no date filtering)
        # We need future labels for the 3-month offset training approach
        labels_sdf = spark.read.parquet(label_store_path)
        print(f"Labels available: {labels_sdf.count()} records")
        
        # Create training dataset with time offset
        # For each feature date, join with labels from feature_date + 3 months
        training_rows = []
        
        # Get unique feature dates
        feature_dates = [r.feature_snapshot_date for r in features_sdf.select("feature_snapshot_date").distinct().collect()]
        label_dates = [r.snapshot_date for r in labels_sdf.select("snapshot_date").distinct().collect()]
        
        print(f"Processing {len(feature_dates)} feature dates...")
        print(f"Available feature dates: {sorted(feature_dates)}")
        print(f"Available label dates: {sorted(label_dates)}")
        
        for feat_date in sorted(feature_dates):
            # Calculate corresponding label date (3 months later)
            label_date = feat_date + relativedelta(months=3)
            
            # Check if we have labels for this date
            if label_date not in label_dates:
                print(f"  {feat_date} -> {label_date}: No labels available for this future date")
                continue
            
            # Get features for this date
            feat_slice = features_sdf.filter(col("feature_snapshot_date") == feat_date)
            
            # Get labels for the corresponding future date
            label_slice = labels_sdf.filter(col("snapshot_date") == label_date)
            
            # Join on Customer_ID
            joined = feat_slice.join(label_slice.select("Customer_ID", "label"), on="Customer_ID", how="inner")
            
            if joined.count() > 0:
                training_rows.append(joined)
                print(f"  {feat_date} -> {label_date}: {joined.count()} training examples")
            else:
                print(f"  {feat_date} -> {label_date}: No matching customers")
        
        if not training_rows:
            # If no 3-month offset data available, try with shorter offsets or no offset
            print("No 3-month offset data found. Trying alternative approaches...")
            
            # Try 1-month offset
            for feat_date in sorted(feature_dates):
                label_date = feat_date + relativedelta(months=1)
                if label_date not in label_dates:
                    continue
                    
                feat_slice = features_sdf.filter(col("feature_snapshot_date") == feat_date)
                label_slice = labels_sdf.filter(col("snapshot_date") == label_date)
                joined = feat_slice.join(label_slice.select("Customer_ID", "label"), on="Customer_ID", how="inner")
                
                if joined.count() > 0:
                    training_rows.append(joined)
                    print(f"  {feat_date} -> {label_date}: {joined.count()} training examples (1-month offset)")
            
            # If still no data, try same-date join
            if not training_rows:
                for feat_date in sorted(feature_dates):
                    if feat_date not in label_dates:
                        continue
                        
                    feat_slice = features_sdf.filter(col("feature_snapshot_date") == feat_date)
                    label_slice = labels_sdf.filter(col("snapshot_date") == feat_date)
                    joined = feat_slice.join(label_slice.select("Customer_ID", "label"), on="Customer_ID", how="inner")
                    
                    if joined.count() > 0:
                        training_rows.append(joined)
                        print(f"  {feat_date} -> {feat_date}: {joined.count()} training examples (same-date join)")
        
        if not training_rows:
            raise ValueError("No valid training examples found with any time offset approach")
        
        # Combine all training slices
        training_dataset = training_rows[0]
        for additional_slice in training_rows[1:]:
            training_dataset = training_dataset.unionByName(additional_slice)
        
        # Convert to pandas
        training_pdf = training_dataset.toPandas()
        
        print(f"Training data created successfully: {len(training_pdf)} records")
        print(f"Label distribution: {training_pdf['label'].value_counts().to_dict()}")
        print(f"Feature snapshot dates used: {sorted(training_pdf['feature_snapshot_date'].unique())}")
        
        return training_pdf
        
    except Exception as e:
        print(f"ERROR loading training data: {str(e)}")
        raise e

def prepare_features(training_data):
    """Prepare features for model training"""
    # Exclude non-feature columns
    exclude_cols = ['Customer_ID', 'feature_snapshot_date', 'label', 'target']
    
    # Get feature columns (numeric only)
    feature_cols = [col for col in training_data.columns 
                   if col not in exclude_cols and training_data[col].dtype in ['int64', 'float64']]
    
    X = training_data[feature_cols].fillna(0)
    
    # Get target variable (try different possible label column names)
    target_col = None
    for col_name in ['target', 'label', 'default_flag', 'bad_flag']:
        if col_name in training_data.columns:
            target_col = col_name
            break
    
    if target_col is None:
        raise ValueError("No target variable found in training data")
    
    y = training_data[target_col]
    
    print(f"Features prepared: {len(feature_cols)} features, target: {target_col}")
    return X, y, feature_cols

def train_model(model_name, X_train, X_test, y_train, y_test, X_oot_periods, y_oot_periods):
    """Train individual model with comprehensive OOT evaluation across multiple periods"""
    print(f"Training {model_name}...")
    
    # Split training data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    if model_name == "xgboost":
        # XGBoost with early stopping
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": 88,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "gamma": 0.5,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "verbosity": 0
        }
        
        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        best_iter = model.best_iteration
        
        # Get predictions for train/test
        train_pred_proba = model.predict(xgb.DMatrix(X_train), iteration_range=(0, best_iter))
        test_pred_proba = model.predict(xgb.DMatrix(X_test), iteration_range=(0, best_iter))
        
        # Get predictions for individual OOT periods
        oot_individual_scores = {}
        oot_combined_preds = []
        oot_combined_labels = []
        
        for period_name, X_oot_period in X_oot_periods.items():
            y_oot_period = y_oot_periods[period_name]
            dtest_period = xgb.DMatrix(X_oot_period)
            oot_pred_period = model.predict(dtest_period, iteration_range=(0, best_iter))
            oot_score = roc_auc_score(y_oot_period, oot_pred_period)
            oot_individual_scores[period_name] = oot_score
            
            oot_combined_preds.extend(oot_pred_period)
            oot_combined_labels.extend(y_oot_period)
        
        oot_combined_auc = roc_auc_score(oot_combined_labels, oot_combined_preds)
        
    elif model_name == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Please install lightgbm or use a different model.")
        
        model = lgb.LGBMClassifier(
            objective='binary',
            random_state=88,
            n_jobs=-1,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=5.0,
            n_estimators=2000,
            verbose=-1
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Get predictions
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Get predictions for individual OOT periods
        oot_individual_scores = {}
        oot_combined_preds = []
        oot_combined_labels = []
        
        for period_name, X_oot_period in X_oot_periods.items():
            y_oot_period = y_oot_periods[period_name]
            oot_pred_period = model.predict_proba(X_oot_period)[:, 1]
            oot_score = roc_auc_score(y_oot_period, oot_pred_period)
            oot_individual_scores[period_name] = oot_score
            
            oot_combined_preds.extend(oot_pred_period)
            oot_combined_labels.extend(y_oot_period)
        
        oot_combined_auc = roc_auc_score(oot_combined_labels, oot_combined_preds)
        
    elif model_name == "catboost":
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available. Please install catboost or use a different model.")
        
        model = cb.CatBoostClassifier(
            loss_function='Logloss',
            random_seed=88,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=10,
            subsample=0.7,
            rsm=0.7,
            iterations=2000,
            verbose=False
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            use_best_model=True
        )
        
        # Get predictions
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Get predictions for individual OOT periods
        oot_individual_scores = {}
        oot_combined_preds = []
        oot_combined_labels = []
        
        for period_name, X_oot_period in X_oot_periods.items():
            y_oot_period = y_oot_periods[period_name]
            oot_pred_period = model.predict_proba(X_oot_period)[:, 1]
            oot_score = roc_auc_score(y_oot_period, oot_pred_period)
            oot_individual_scores[period_name] = oot_score
            
            oot_combined_preds.extend(oot_pred_period)
            oot_combined_labels.extend(y_oot_period)
        
        oot_combined_auc = roc_auc_score(oot_combined_labels, oot_combined_preds)
        
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: xgboost, lightgbm, catboost")
    
    # Calculate main AUC scores
    train_auc = roc_auc_score(y_train, train_pred_proba)
    test_auc = roc_auc_score(y_test, test_pred_proba)
    
    # Calculate OOT stability metrics
    oot_scores = list(oot_individual_scores.values())
    oot_avg = np.mean(oot_scores)
    oot_std = np.std(oot_scores)
    oot_cv = oot_std / oot_avg if oot_avg > 0 else 0  # Coefficient of variation
    
    print(f"{model_name} - Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, OOT Combined: {oot_combined_auc:.4f}")
    print(f"  OOT Individual: {[f'{k}: {v:.4f}' for k, v in oot_individual_scores.items()]}")
    print(f"  OOT Stability: Avg={oot_avg:.4f} (±{oot_std:.4f}), CV={oot_cv:.4f}")
    
    return model, {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'oot_combined_auc': oot_combined_auc,
        'oot_individual_scores': oot_individual_scores,
        'oot_avg': oot_avg,
        'oot_std': oot_std,
        'oot_cv': oot_cv,
        'model_name': model_name
    }

def train_and_evaluate_models(training_data, config_name, models_to_train):
    """Train multiple models and select the best one based on OOT performance across multiple periods"""
    print(f"Starting model training for {config_name}")
    
    # Prepare data
    X, y, feature_cols = prepare_features(training_data)
    
    # Create 3 separate OOT periods (like in the notebook)
    training_data_sorted = training_data.sort_values('feature_snapshot_date')
    
    # Split into train/test (first 60%) and OOT (last 40%)
    train_test_size = int(len(training_data_sorted) * 0.6)
    train_test_data = training_data_sorted.iloc[:train_test_size]
    oot_full_data = training_data_sorted.iloc[train_test_size:]
    
    # Prepare train/test data
    X_full, y_full, _ = prepare_features(train_test_data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.25, random_state=42, stratify=y_full
    )
    
    # Create 3 OOT periods from the remaining data
    oot_period_size = len(oot_full_data) // 3
    
    oot_periods = {}
    X_oot_periods = {}
    y_oot_periods = {}
    
    for i in range(3):
        period_name = f"OOT_{i+1}"
        start_idx = i * oot_period_size
        end_idx = (i + 1) * oot_period_size if i < 2 else len(oot_full_data)
        
        oot_period_data = oot_full_data.iloc[start_idx:end_idx]
        X_oot_period, y_oot_period, _ = prepare_features(oot_period_data)
        
        oot_periods[period_name] = oot_period_data
        X_oot_periods[period_name] = X_oot_period
        y_oot_periods[period_name] = y_oot_period
        
        print(f"{period_name}: {len(oot_period_data)} records")
    
    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}, Total OOT={sum(len(x) for x in X_oot_periods.values())}")
    
    # Validate model list - only tree-based models
    supported_models = ['xgboost', 'lightgbm', 'catboost']
    models_to_train = [m for m in models_to_train if m in supported_models]
    
    if not models_to_train:
        models_to_train = supported_models  # Default to all supported models
    
    # Remove catboost if not available
    if 'catboost' in models_to_train and not CATBOOST_AVAILABLE:
        print("WARNING: CatBoost not available, removing from training list")
        models_to_train = [m for m in models_to_train if m != 'catboost']
    
    # Remove lightgbm if not available
    if 'lightgbm' in models_to_train and not LIGHTGBM_AVAILABLE:
        print("WARNING: LightGBM not available, removing from training list")
        models_to_train = [m for m in models_to_train if m != 'lightgbm']
    
    if not models_to_train:
        raise ValueError(f"No valid models to train. Supported models: {supported_models}")
    
    # Train models
    model_results = {}
    
    for model_name in models_to_train:
        try:
            model, metrics = train_model(model_name, X_train, X_test, y_train, y_test, X_oot_periods, y_oot_periods)
            model_results[model_name] = {
                'model': model,
                'metrics': metrics
            }
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    if not model_results:
        raise ValueError("No models were successfully trained")
    
    # Select best model based on average OOT AUC across periods
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['metrics']['oot_avg'])
    best_model_info = model_results[best_model_name]
    best_metrics = best_model_info['metrics']
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    for name, result in model_results.items():
        m = result['metrics']
        print(f"{name:12} | Train: {m['train_auc']:.4f} | Test: {m['test_auc']:.4f} | OOT Avg: {m['oot_avg']:.4f} (±{m['oot_std']:.4f})")
        print(f"{'':12} | Individual OOT: {[f'{k}: {v:.4f}' for k, v in m['oot_individual_scores'].items()]}")
    
    print(f"\n🏆 Best model: {best_model_name} with OOT Average AUC: {best_metrics['oot_avg']:.4f}")
    print(f"   OOT Stability (CV): {best_metrics['oot_cv']:.4f}")
    
    # Create comprehensive model artifact with multi-period OOT performance
    model_artifact = {
        'model': best_model_info['model'],
        'model_name': best_model_name,
        'model_version': config_name,
        'feature_columns': feature_cols,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_records': len(training_data),
        
        # Performance metrics
        'performance': {
            'train_auc': best_metrics['train_auc'],
            'test_auc': best_metrics['test_auc'],
            'oot_combined_auc': best_metrics['oot_combined_auc'],
            'oot_avg': best_metrics['oot_avg'],
            'oot_std': best_metrics['oot_std'],
            'oot_cv': best_metrics['oot_cv'],
            'oot_individual_scores': best_metrics['oot_individual_scores'],
            
            'train_gini': round(2 * best_metrics['train_auc'] - 1, 4),
            'test_gini': round(2 * best_metrics['test_auc'] - 1, 4),
            'oot_combined_gini': round(2 * best_metrics['oot_combined_auc'] - 1, 4),
            'oot_avg_gini': round(2 * best_metrics['oot_avg'] - 1, 4)
        },
        
        # Data split information
        'data_split': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'oot_periods': {name: len(X_oot_periods[name]) for name in X_oot_periods.keys()},
            'train_positive_rate': round(y_train.mean(), 4),
            'test_positive_rate': round(y_test.mean(), 4),
            'oot_positive_rates': {name: round(y_oot_periods[name].mean(), 4) for name in y_oot_periods.keys()}
        },
        
        # All model results for comparison
        'all_model_results': {
            name: result['metrics'] for name, result in model_results.items()
        },
        
        # OOT stability analysis
        'oot_stability': {
            'coefficient_of_variation': best_metrics['oot_cv'],
            'individual_scores': best_metrics['oot_individual_scores'],
            'stability_assessment': 'Stable' if best_metrics['oot_cv'] < 0.1 else 'Moderate' if best_metrics['oot_cv'] < 0.2 else 'Unstable'
        }
    }
    
    # Save model
    model_filename = f"{config_name}.pkl"
    model_path = os.path.join("model_bank", model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_artifact, f)
    
    print(f"Model saved: {model_path}")
    
    return model_artifact

def run_data_pipeline(spark):
    """Run the complete data pipeline"""
    print("Starting data pipeline...")
    
    # Bronze
    ingest_bronze_tables(spark)
    print("Bronze layer complete!")

    # Silver
    clean_financials_table(spark)
    clean_attributes_table(spark)
    clean_clickstream_table(spark)
    clean_loans_table(spark)
    print("Silver layer complete!")

    # Gold
    build_label_store(spark, dpd_cutoff=30, mob_cutoff=3)
    build_feature_store(spark, dpd_cutoff=30, mob_cutoff=3)
    print("Gold layer complete!")

def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline")
    parser.add_argument("--snapshotdate", type=str, help="YYYY-MM-DD", default=None)
    parser.add_argument("--config_name", type=str, help="Model configuration name", default=None)
    parser.add_argument("--train_models", type=str, help="Comma-separated list of models to train", default="xgboost,lightgbm,catboost")
    parser.add_argument("--mode", type=str, choices=["data_pipeline", "train_evaluate", "full"], default="full")
    
    args = parser.parse_args()
    
    print(f"Starting MLOps Pipeline with arguments:")
    print(f"  Mode: {args.mode}")
    print(f"  Snapshot Date: {args.snapshotdate}")
    print(f"  Config Name: {args.config_name}")
    print(f"  Train Models: {args.train_models}")
    
    # Initialize
    spark = None
    try:
        spark = create_spark_session()
        init_datamart()
        
        if args.mode in ["data_pipeline", "full"]:
            print("Running data pipeline...")
            run_data_pipeline(spark)
            print("Data pipeline completed successfully!")
        
        if args.mode in ["train_evaluate", "full"]:
            if not args.snapshotdate or not args.config_name:
                raise ValueError("snapshotdate and config_name required for model training")
            
            print("Starting model training phase...")
            
            # Check if gold tables exist before training
            feature_store_path = "datamart/gold/feature_store"
            label_store_path = "datamart/gold/label_store"
            
            if not os.path.exists(feature_store_path):
                raise FileNotFoundError(f"Feature store not found: {feature_store_path}. Run data pipeline first.")
            
            if not os.path.exists(label_store_path):
                raise FileNotFoundError(f"Label store not found: {label_store_path}. Run data pipeline first.")
            
            # Load training data
            print("Loading training data...")
            training_data = load_training_data(spark, args.snapshotdate)
            
            # Parse models to train
            models_to_train = [m.strip() for m in args.train_models.split(",")]
            print(f"Models to train: {models_to_train}")
            
            # Validate model availability  
            supported_models = ['xgboost', 'lightgbm', 'catboost']
            invalid_models = [m for m in models_to_train if m not in supported_models]
            
            if invalid_models:
                print(f"WARNING: Unsupported models removed: {invalid_models}")
                models_to_train = [m for m in models_to_train if m in supported_models]
            
            # Remove catboost if not available
            if 'catboost' in models_to_train and not CATBOOST_AVAILABLE:
                print("WARNING: CatBoost not available, removing from training list")
                models_to_train = [m for m in models_to_train if m != 'catboost']
            
            # Remove lightgbm if not available
            if 'lightgbm' in models_to_train and not LIGHTGBM_AVAILABLE:
                print("WARNING: LightGBM not available, removing from training list")
                models_to_train = [m for m in models_to_train if m != 'lightgbm']
            
            if not models_to_train:
                raise ValueError(f"No valid models to train. Supported models: {supported_models}")
            
            # Train and evaluate models
            print("Starting model training and evaluation...")
            model_artifact = train_and_evaluate_models(training_data, args.config_name, models_to_train)
            
            print(f"Model training completed successfully!")
            print(f"Best model: {model_artifact['model_name']}")
            print(f"Best OOT Average AUC: {model_artifact['performance']['oot_avg']:.4f}")
            print(f"OOT Stability (CV): {model_artifact['performance']['oot_cv']:.4f}")
            print(f"Model saved as: {args.config_name}.pkl")
            print(f"OOT Combined Gini: {model_artifact['performance']['oot_combined_gini']:.4f}")
            
    except FileNotFoundError as e:
        print(f"FILE ERROR: {e}")
        print("SOLUTION: Ensure data pipeline has been run first to create gold tables")
        raise e
    except ValueError as e:
        print(f"VALIDATION ERROR: {e}")
        raise e
    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        if spark:
            spark.stop()
            print("Spark session stopped")

if __name__ == "__main__":
    main()
