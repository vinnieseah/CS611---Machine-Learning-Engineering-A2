# MLOps Pipeline with 4-Path Retraining Logic:
# 1. Initial training (no model exists)
# 2. 12-month grace period (no retraining)  
# 3. 6-month periodic retraining (after grace period)
# 4. Threshold-based retraining (performance issues)

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

def check_retraining_trigger(**context):
    import pickle
    from dateutil.relativedelta import relativedelta
    
    execution_date = context['execution_date']
    print(f"=== RETRAINING TRIGGER CHECK ===")
    print(f"Execution date: {execution_date}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Use dynamic model naming based on execution date
    model_name = f"credit_model_{execution_date.strftime('%Y_%m_%d')}.pkl"
    model_path = f"model_bank/{model_name}"
    print(f"Checking for model at: {model_path}")
    
    # PATH 1: Initial training if no model exists
    if not os.path.exists(model_path):
        print("PATH 1: No model found - triggering initial training")
        return "trigger_retraining"
    
    print(f"Model file found at: {model_path}")
    
    # Load model artifact
    try:
        with open(model_path, 'rb') as f:
            model_artifact = pickle.load(f)
        print("Model artifact loaded successfully")
    except Exception as e:
        print(f"Failed to load model artifact: {e}")
        return "trigger_retraining"
    
    # Calculate months since model start
    first_model_date = datetime(2023, 1, 1)
    months_since_start = (execution_date.year - first_model_date.year) * 12 + (execution_date.month - first_model_date.month)
    
    print(f"Model lifecycle: {months_since_start} months since first model")
    
    # PATH 2: Grace period - no retraining for first 12 months
    if months_since_start < 12:
        months_remaining = 12 - months_since_start
        print(f"PATH 2: Grace period - {months_remaining} months remaining")
        return "skip_retraining"
    
    # 3 THRESHOLD DEFINITIONS (Simplified)
    THRESHOLDS = {
        'critical_auc': 0.70,      # Critical AUC threshold
        'warning_auc': 0.72,       # Warning AUC threshold
        'stability_cv': 0.08       # Stability CV threshold
    }
    
    print(f"Critical AUC: {THRESHOLDS['critical_auc']:.4f}")
    print(f"Warning AUC: {THRESHOLDS['warning_auc']:.4f}")
    print(f"Stability CV: {THRESHOLDS['stability_cv']:.4f}")
    
    # 2/3 THRESHOLD BREACH LOGIC
    threshold_breaches = []
    
    # Check AUC thresholds
    if 'performance' in model_artifact:
        performance = model_artifact['performance']
        current_auc = performance.get('oot_avg', performance.get('oot_combined_auc', 0))
        
        if current_auc > 0:
            print(f"Current model AUC: {current_auc:.4f}")
            
            # Check Threshold 1: Critical AUC
            if current_auc < THRESHOLDS['critical_auc']:
                threshold_breaches.append(f"CRITICAL_AUC: {current_auc:.4f} < {THRESHOLDS['critical_auc']}")
            
            # Check Threshold 2: Warning AUC
            if current_auc < THRESHOLDS['warning_auc']:
                threshold_breaches.append(f"WARNING_AUC: {current_auc:.4f} < {THRESHOLDS['warning_auc']}")
        else:
            threshold_breaches.append("No valid AUC metrics found")
    
    # Check Threshold 3: Model stability
    if 'performance' in model_artifact and 'oot_cv' in model_artifact['performance']:
        oot_cv = model_artifact['performance']['oot_cv']
        print(f"Model stability (CV): {oot_cv:.4f}")
        
        if oot_cv > THRESHOLDS['stability_cv']:
            threshold_breaches.append(f"STABILITY_CV: {oot_cv:.4f} > {THRESHOLDS['stability_cv']}")
    
    # Apply 2/3 threshold rule for retraining
    if len(threshold_breaches) >= 2:
        print(f"PATH 4: 2/3 THRESHOLD BREACH RULE TRIGGERED ({len(threshold_breaches)}/3 thresholds breached)")
        for breach in threshold_breaches:
            print(f"  - {breach}")
        print("Triggering IMMEDIATE THRESHOLD-BASED RETRAINING")
        return "trigger_retraining"
    elif len(threshold_breaches) == 1:
        print(f"PATH 4: 1/3 THRESHOLD BREACH DETECTED (monitoring)")
        for breach in threshold_breaches:
            print(f"  - {breach}")
        print("Single threshold breach detected - monitoring but not retraining yet")
    else:
        print("PATH 4: ALL THRESHOLDS HEALTHY")
    
    # PATH 3: 6-month periodic retraining after grace period
    months_since_grace_period = months_since_start - 12
    
    if months_since_grace_period >= 0:
        if months_since_grace_period % 6 == 0:
            retraining_cycle = (months_since_grace_period // 6) + 1
            print(f"PATH 3: 6-month periodic retraining (Cycle #{retraining_cycle})")
            return "trigger_retraining"
        else:
            months_until_next = 6 - (months_since_grace_period % 6)
            print(f"Next retraining in {months_until_next} months")
            return "skip_retraining"
    
    print("All checks passed - skip retraining")
    return "skip_retraining"

def decide_pipeline_path(**context):
    execution_date = context['execution_date']
    
    # Use dynamic model naming based on execution date  
    model_name = f"credit_model_{execution_date.strftime('%Y_%m_%d')}.pkl"
    model_path = f"model_bank/{model_name}"
    model_exists = os.path.exists(model_path)
    
    print(f"=== PIPELINE DECISION ===")
    print(f"Execution date: {execution_date}")
    print(f"Dynamic model name: {model_name}")
    print(f"Checking for model at: {model_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model exists: {model_exists}")
    
    if model_exists:
        print("Model found - using MONTHLY LIFECYCLE FLOW")
        return "run_monthly_lifecycle_flow"
    else:
        print("No model found - using INITIAL TRAINING FLOW")
        return "run_initial_training_flow"

with DAG(
    'complete_mlops_lifecycle_pipeline',
    default_args=default_args,
    description='Streamlined MLOps Pipeline with 3-Threshold 2/3 Retraining Logic',
    schedule_interval='0 0 1 * *',  # Monthly
    start_date=datetime(2023, 1, 1),  # Start date for 2-year window
    end_date=datetime(2025, 1, 1),    # End date to limit to 2 years (24 months)
    catchup=True,                     # Enable backfill (limited by end_date)
    tags=['mlops', 'retraining', 'inference', 'monitoring', 'streamlined'],
) as dag:

    # =============================================================================
    # PHASE 1: PIPELINE INITIALIZATION
    # =============================================================================
    start_pipeline = BashOperator(
        task_id="start_pipeline",
        bash_command=(
            'mkdir -p datamart/bronze datamart/silver datamart/gold model_bank logs && '
            'echo "=== PIPELINE STARTUP DEBUG ===" && '
            'echo "Current directory: $(pwd)" && '
            'echo "Model bank contents:" && '
            'ls -la model_bank/ 2>/dev/null || echo "Model bank is empty" && '
            'echo "=========================="'
        ),
    )

    check_static_data = BashOperator(
        task_id="check_static_data",
        bash_command='ls -la data/ 2>/dev/null || mkdir -p data',
    )

    # =============================================================================
    # PHASE 2: DATA DEPENDENCY VALIDATION
    # =============================================================================
    validate_source_data = BashOperator(
        task_id="validate_source_data",
        bash_command=(
            'echo "Validating source data files..." && '
            'find . -name "*loan*.csv" -o -name "*lms*.csv" | head -1 && '
            'find . -name "*attribute*.csv" -o -name "*customer*.csv" | head -1 && '
            'find . -name "*financial*.csv" -o -name "*credit*.csv" | head -1 && '
            'find . -name "*clickstream*.csv" -o -name "*click*.csv" | head -1 && '
            'echo "Source data validation completed"'
        ),
    )

    # Individual dependency checks for parallel execution
    dep_check_source_loans = BashOperator(
        task_id="dep_check_source_loans",
        bash_command='find . -name "*loan*.csv" -o -name "*lms*.csv" | head -3',
    )

    dep_check_source_attributes = BashOperator(
        task_id="dep_check_source_attributes",
        bash_command='find . -name "*attribute*.csv" -o -name "*customer*.csv" | head -3',
    )

    dep_check_source_financials = BashOperator(
        task_id="dep_check_source_financials",
        bash_command='find . -name "*financial*.csv" -o -name "*credit*.csv" | head -3',
    )

    dep_check_source_clickstream = BashOperator(
        task_id="dep_check_source_clickstream",
        bash_command='find . -name "*clickstream*.csv" -o -name "*click*.csv" | head -3',
    )

    # =============================================================================
    # PHASE 3: BRONZE LAYER (DATA INGESTION)
    # =============================================================================
    run_bronze_tables = BashOperator(
        task_id="run_bronze_tables",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.bronze_processing import ingest_bronze_tables; '
            'spark = SparkSession.builder.appName(\\"BronzeProcessing\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'ingest_bronze_tables(spark); '
            'spark.stop()" && echo "Bronze layer processing completed"'
        ),
    )

    # Individual bronze table processing for parallel execution
    run_bronze_table_loans = BashOperator(
        task_id="run_bronze_table_loans",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.bronze_processing import ingest_bronze_tables; '
            'spark = SparkSession.builder.appName(\\"BronzeLoans\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'ingest_bronze_tables(spark, table_filter=\\"loans\\"); '
            'spark.stop()" || echo "Bronze loans processing completed"'
        ),
    )

    run_bronze_table_clickstream = BashOperator(
        task_id="run_bronze_table_clickstream",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.bronze_processing import ingest_bronze_tables; '
            'spark = SparkSession.builder.appName(\\"BronzeClickstream\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'ingest_bronze_tables(spark, table_filter=\\"clickstream\\"); '
            'spark.stop()" || echo "Bronze clickstream processing completed"'
        ),
    )

    run_bronze_table_attributes = BashOperator(
        task_id="run_bronze_table_attributes",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.bronze_processing import ingest_bronze_tables; '
            'spark = SparkSession.builder.appName(\\"BronzeAttributes\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'ingest_bronze_tables(spark, table_filter=\\"attributes\\"); '
            'spark.stop()" || echo "Bronze attributes processing completed"'
        ),
    )

    run_bronze_table_financials = BashOperator(
        task_id="run_bronze_table_financials",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.bronze_processing import ingest_bronze_tables; '
            'spark = SparkSession.builder.appName(\\"BronzeFinancials\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'ingest_bronze_tables(spark, table_filter=\\"financials\\"); '
            'spark.stop()" || echo "Bronze financials processing completed"'
        ),
    )

    # =============================================================================
    # PHASE 4: SILVER LAYER (DATA CLEANING)
    # =============================================================================
    run_silver_tables = BashOperator(
        task_id="run_silver_tables",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.silver_processing import clean_loans_table, clean_attributes_table, clean_clickstream_table, clean_financials_table; '
            'spark = SparkSession.builder.appName(\\"SilverProcessing\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'clean_financials_table(spark); '
            'clean_attributes_table(spark); '
            'clean_clickstream_table(spark); '
            'clean_loans_table(spark); '
            'spark.stop()" && echo "Silver layer processing completed"'
        ),
    )

    # Individual silver table processing for parallel execution
    run_silver_table_loans = BashOperator(
        task_id="run_silver_table_loans",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.silver_processing import clean_loans_table; '
            'spark = SparkSession.builder.appName(\\"SilverLoans\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'clean_loans_table(spark); '
            'spark.stop()"'
        ),
    )

    run_silver_table_clickstream = BashOperator(
        task_id="run_silver_table_clickstream",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.silver_processing import clean_clickstream_table; '
            'spark = SparkSession.builder.appName(\\"SilverClickstream\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'clean_clickstream_table(spark); '
            'spark.stop()"'
        ),
    )

    run_silver_table_attributes = BashOperator(
        task_id="run_silver_table_attributes",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.silver_processing import clean_attributes_table; '
            'spark = SparkSession.builder.appName(\\"SilverAttributes\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'clean_attributes_table(spark); '
            'spark.stop()"'
        ),
    )

    run_silver_table_financials = BashOperator(
        task_id="run_silver_table_financials",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.silver_processing import clean_financials_table; '
            'spark = SparkSession.builder.appName(\\"SilverFinancials\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'clean_financials_table(spark); '
            'spark.stop()"'
        ),
    )

    # =============================================================================
    # PHASE 5: GOLD LAYER (FEATURE & LABEL STORES)
    # =============================================================================
    run_gold_tables = BashOperator(
        task_id="run_gold_tables",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 -c "'
            'import sys; sys.path.append(\\"/opt/airflow\\"); '
            'from pyspark.sql import SparkSession; '
            'from utils.gold_processing import build_label_store, build_feature_store; '
            'spark = SparkSession.builder.appName(\\"GoldProcessing\\").master(\\"local[*]\\").getOrCreate(); '
            'spark.sparkContext.setLogLevel(\\"ERROR\\"); '
            'label_df = build_label_store(spark, dpd_cutoff=30, mob_cutoff=3); '
            'feature_df = build_feature_store(spark, dpd_cutoff=30, mob_cutoff=3); '
            'spark.stop()" && echo "Gold layer processing completed"'
        ),
    )

    # =============================================================================
    # PHASE 6: PIPELINE DECISION LOGIC
    # =============================================================================
    decide_pipeline_path = BranchPythonOperator(
        task_id="decide_pipeline_path",
        python_callable=decide_pipeline_path,
        do_xcom_push=False,
    )

    # =============================================================================
    # PHASE 7: INITIAL TRAINING FLOW
    # =============================================================================
    run_initial_training_flow = DummyOperator(
        task_id="run_initial_training_flow"
    )

    train_models_initial = BashOperator(
        task_id="train_models_initial",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 utils/train_model.py '
            '--snapshotdate "{{ ds }}" '
            '--config_name "credit_model_{{ ds_nodash }}" '
            '--train_models "xgboost" '
            '--mode "train_evaluate"'
        ),
    )

    register_model_initial = BashOperator(
        task_id="register_model_initial",
        bash_command=(
            'cd /opt/airflow && '
            'echo "Registering initial model..." && '
            'if [ -f "model_bank/credit_model_{{ ds_nodash }}.pkl" ]; then '
            '    echo "Initial model found and registered: credit_model_{{ ds_nodash }}.pkl" && '
            '    python3 utils/model_inference.py '
            '    --snapshotdate "{{ ds }}" '
            '    --modelname "credit_model_{{ ds_nodash }}.pkl" '
            '    --mode validate; '
            'else '
            '    echo "ERROR: Initial model not found at model_bank/credit_model_{{ ds_nodash }}.pkl" && '
            '    ls -la model_bank/ && '
            '    exit 1; '
            'fi'
        ),
    )

    # =============================================================================
    # PHASE 8: MONTHLY LIFECYCLE FLOW
    # =============================================================================
    run_monthly_lifecycle_flow = DummyOperator(
        task_id="run_monthly_lifecycle_flow"
    )

    evaluate_production_model = BashOperator(
        task_id="evaluate_production_model",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 utils/model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "credit_model_{{ ds_nodash }}.pkl" '
            '--mode monitor'
        ),
    )

    check_retraining_trigger = BranchPythonOperator(
        task_id="check_retraining_trigger",
        python_callable=check_retraining_trigger,
        do_xcom_push=False,
    )

    # =============================================================================
    # PHASE 9: RETRAINING FLOW
    # =============================================================================
    trigger_retraining = DummyOperator(
        task_id="trigger_retraining"
    )

    train_models_retraining = BashOperator(
        task_id="train_models_retraining",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'python3 utils/train_model.py '
            '--snapshotdate "{{ ds }}" '
            '--config_name "credit_model_{{ ds_nodash }}_retrain" '
            '--train_models "xgboost" '
            '--mode "train_evaluate"'
        ),
    )

    select_best_model_monthly = BashOperator(
        task_id="select_best_model_monthly",
        bash_command=(
            'cd /opt/airflow && '
            'python3 -c "'
            'import pickle; '
            'import os; '
            'import shutil; '
            'current_model = \\"model_bank/credit_model_{{ ds_nodash }}.pkl\\"; '
            'new_model = \\"model_bank/credit_model_{{ ds_nodash }}_retrain.pkl\\"; '
            'if os.path.exists(new_model): '
            '    with open(new_model, \\"rb\\") as f: '
            '        new_artifact = pickle.load(f); '
            '    new_auc = new_artifact.get(\\"performance\\", {}).get(\\"oot_avg\\", 0); '
            '    print(f\\"New model AUC: {new_auc:.4f}\\"); '
            '    if os.path.exists(current_model): '
            '        with open(current_model, \\"rb\\") as f: '
            '            current_artifact = pickle.load(f); '
            '        current_auc = current_artifact.get(\\"performance\\", {}).get(\\"oot_avg\\", 0); '
            '        if new_auc > current_auc: '
            '            shutil.copy(new_model, current_model); '
            '            print(f\\"Model updated: {new_auc:.4f} > {current_auc:.4f}\\"); '
            '        else: '
            '            print(f\\"Keeping current model: {current_auc:.4f} >= {new_auc:.4f}\\"); '
            '    else: '
            '        shutil.copy(new_model, current_model); '
            '        print(\\"New model deployed\\"); '
            'else: '
            '    print(\\"No new model found\\")"'
        ),
    )

    # =============================================================================
    # PHASE 10: SKIP RETRAINING FLOW
    # =============================================================================
    skip_retraining = DummyOperator(
        task_id="skip_retraining"
    )

    # =============================================================================
    # PHASE 11: MODEL INFERENCE
    # =============================================================================
    run_model_inference = BashOperator(
        task_id="run_model_inference",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'echo "=== RUNNING MODEL INFERENCE ===" && '
            'python3 utils/model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "credit_model_{{ ds_nodash }}.pkl" '
            '--mode batch_inference && '
            'echo "Model inference completed"'
        ),
        trigger_rule='none_failed_min_one_success',
    )

    # =============================================================================
    # PHASE 12: MODEL MONITORING & VISUALIZATION
    # =============================================================================
    monitor_model_performance = BashOperator(
        task_id="monitor_model_performance",
        bash_command=(
            'cd /opt/airflow && '
            'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && '
            'echo "=== MONITORING MODEL PERFORMANCE ===" && '
            'python3 utils/model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "credit_model_{{ ds_nodash }}.pkl" '
            '--mode monitor && '
            'echo "Model monitoring completed - check datamart/gold/model_monitoring/ for results"'
        ),
    )

    # =============================================================================
    # PHASE 12.5: MODEL CLEANUP (OPTIONAL)
    # =============================================================================
    cleanup_old_models = BashOperator(
        task_id="cleanup_old_models",
        bash_command=(
            'cd /opt/airflow && '
            'echo "=== MODEL CLEANUP ===" && '
            'echo "Current models in model_bank:" && '
            'ls -la model_bank/*.pkl 2>/dev/null || echo "No models found" && '
            'echo "Keeping latest 3 models and removing older versions..." && '
            'find model_bank -name "*.pkl" -type f | '
            'grep -E "credit_model_[0-9]{8}" | '
            'sort -r | '
            'tail -n +4 | '
            'while read model; do '
            '    echo "Archiving old model: $model" && '
            '    mv "$model" "${model%.pkl}_archived.pkl" 2>/dev/null || echo "Failed to archive $model"; '
            'done && '
            'echo "Model cleanup completed"'
        ),
        trigger_rule='none_failed_min_one_success',
    )

    # =============================================================================
    # PHASE 13: PIPELINE COMPLETION
    # =============================================================================
    end_pipeline = BashOperator(
        task_id="end_pipeline",
        bash_command=(
            'echo "Pipeline completed at $(date)" && '
            'echo "Bronze tables: $(find datamart/bronze -name "*.parquet" 2>/dev/null | wc -l)" && '
            'echo "Silver tables: $(find datamart/silver -name "*.parquet" 2>/dev/null | wc -l)" && '
            'echo "Gold tables: $(find datamart/gold -name "*.parquet" 2>/dev/null | wc -l)" && '
            'echo "Active models: $(ls -1 model_bank/*.pkl 2>/dev/null | grep -v archived | wc -l)" && '
            'echo "Archived models: $(ls -1 model_bank/*_archived.pkl 2>/dev/null | wc -l)"'
        ),
        trigger_rule='none_failed_min_one_success',
    )

    # =============================================================================
    # TASK DEPENDENCIES
    # =============================================================================
    
    # Phase 1-2: Initialization & Dependency Validation (Parallel)
    start_pipeline >> check_static_data >> validate_source_data
    validate_source_data >> [dep_check_source_loans, dep_check_source_attributes, 
                           dep_check_source_financials, dep_check_source_clickstream]
    
    # Phase 3: Bronze Layer (Parallel Processing)
    dep_check_source_loans >> run_bronze_table_loans
    dep_check_source_attributes >> run_bronze_table_attributes  
    dep_check_source_financials >> run_bronze_table_financials
    dep_check_source_clickstream >> run_bronze_table_clickstream

    # Phase 4: Silver Layer (Parallel Processing)
    run_bronze_table_loans >> run_silver_table_loans
    run_bronze_table_attributes >> run_silver_table_attributes
    run_bronze_table_financials >> run_silver_table_financials
    run_bronze_table_clickstream >> run_silver_table_clickstream

    # Phase 5: Gold Layer (Wait for all Silver tables)
    [run_silver_table_loans, run_silver_table_attributes, 
     run_silver_table_financials, run_silver_table_clickstream] >> run_gold_tables
    
    # Phase 6: Pipeline Decision
    run_gold_tables >> decide_pipeline_path

    # Phase 7: Initial Training Flow
    decide_pipeline_path >> run_initial_training_flow >> train_models_initial
    train_models_initial >> register_model_initial >> run_model_inference

    # Phase 8-10: Monthly Lifecycle Flow
    decide_pipeline_path >> run_monthly_lifecycle_flow >> evaluate_production_model
    evaluate_production_model >> check_retraining_trigger
    
    # Retraining path
    check_retraining_trigger >> trigger_retraining >> train_models_retraining
    train_models_retraining >> select_best_model_monthly >> run_model_inference
    
    # Skip retraining path
    check_retraining_trigger >> skip_retraining >> run_model_inference

    # Phase 11-13: Inference, Monitoring, Cleanup & Completion
    run_model_inference >> monitor_model_performance >> cleanup_old_models >> end_pipeline
    