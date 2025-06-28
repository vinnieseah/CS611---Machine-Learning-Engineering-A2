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
        
        # Load feature store
        features_sdf = spark.read.parquet(feature_store_path)
        features_sdf = features_sdf.filter(col("feature_snapshot_date") <= snapshot_date)
        print(f"Features available: {features_sdf.count()} records")
        
        # Load label store  
        labels_sdf = spark.read.parquet(label_store_path)
        labels_sdf = labels_sdf.filter(col("snapshot_date") <= snapshot_date)
        print(f"Labels available: {labels_sdf.count()} records")
        
        # Create training dataset with time offset
        # For each feature date, join with labels from feature_date + 3 months
        training_rows = []
        
        # Get unique feature dates
        feature_dates = [r.feature_snapshot_date for r in features_sdf.select("feature_snapshot_date").distinct().collect()]
        label_dates = [r.snapshot_date for r in labels_sdf.select("snapshot_date").distinct().collect()]
        
        print(f"Processing {len(feature_dates)} feature dates...")
        
        for feat_date in sorted(feature_dates):
            # Calculate corresponding label date (3 months later)
            label_date = feat_date + relativedelta(months=3)
            
            # Check if we have labels for this date
            if label_date not in label_dates:
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
            raise ValueError("No valid training examples found with 3-month time offset")
        
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

def train_model(model_name, X_train, X_test, y_train, y_test, scaler=None):
    """Train individual model"""
    print(f"Training {model_name}...")
    
    if model_name == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == "xgboost":
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    elif model_name == "catboost":
        if not CATBOOST_AVAILABLE:
            raise ImportError(f"CatBoost not available. Please install catboost or use a different model.")
        model = cb.CatBoostClassifier(random_state=42, verbose=False)
    elif model_name == "neural_network":
        model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Apply scaling for neural network and logistic regression
    if model_name in ["neural_network", "logistic_regression"] and scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"{model_name} - AUC: {auc_score:.4f}")
    
    return model, auc_score

def train_and_evaluate_models(training_data, config_name, models_to_train):
    """Train multiple models and select the best one"""
    print(f"Starting model training for {config_name}")
    
    # Prepare data
    X, y, feature_cols = prepare_features(training_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Train models
    model_results = {}
    
    for model_name in models_to_train:
        try:
            model, auc_score = train_model(model_name, X_train, X_test, y_train, y_test, scaler)
            model_results[model_name] = {
                'model': model,
                'auc_score': auc_score,
                'model_name': model_name
            }
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    if not model_results:
        raise ValueError("No models were successfully trained")
    
    # Select best model
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc_score'])
    best_model_info = model_results[best_model_name]
    
    print(f"Best model: {best_model_name} with AUC: {best_model_info['auc_score']:.4f}")
    
    # Create model artifact
    model_artifact = {
        'model': best_model_info['model'],
        'model_name': best_model_name,
        'auc_score': best_model_info['auc_score'],
        'feature_columns': feature_cols,
        'preprocessing_transformers': {'stdscaler': scaler},
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_records': len(training_data),
        'model_results': {k: v['auc_score'] for k, v in model_results.items()}
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
    parser.add_argument("--train_models", type=str, help="Comma-separated list of models to train", default="logistic_regression,random_forest,xgboost")
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
            if "catboost" in models_to_train and not CATBOOST_AVAILABLE:
                print("WARNING: CatBoost not available, removing from training list")
                models_to_train = [m for m in models_to_train if m != "catboost"]
            
            if not models_to_train:
                raise ValueError("No valid models to train after filtering")
            
            # Train and evaluate models
            print("Starting model training and evaluation...")
            model_artifact = train_and_evaluate_models(training_data, args.config_name, models_to_train)
            
            print(f"Model training completed successfully!")
            print(f"Best model: {model_artifact['model_name']}")
            print(f"Best AUC: {model_artifact['auc_score']:.4f}")
            print(f"Model saved as: {args.config_name}.pkl")
            
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
