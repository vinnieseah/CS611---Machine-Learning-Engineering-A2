from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import glob, re, pickle, os
from dateutil.relativedelta import relativedelta

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

def decide_initial_pipeline_path(**context):
    """If no model exists at all -> train; otherwise skip to inference."""
    models = glob.glob("/opt/airflow/model_bank/credit_model_*.pkl")
    if not models:
        return "train_model"
    return "model_inference"

def check_retraining_trigger(**context):
    """4-path retraining logic:
       1) initial train if no models; 
       2) grace period <12m; 
       3) periodic every 6m; 
       4) threshold-based (2/3 breaches)."""
    execution_date = context['execution_date']
    cur = execution_date.strftime('%Y%m%d')

    all_models = glob.glob("/opt/airflow/model_bank/credit_model_*.pkl")
    active    = [m for m in all_models 
                 if not m.endswith(('_archived.pkl','_retrain.pkl'))]

    # PATH 1
    if not all_models:
        return "retrain_model"

    # only consider those older than today
    prev = []
    for m in active:
        d = re.search(r'credit_model_(\d{8})\.pkl', m)
        if d and d.group(1) < cur:
            prev.append((m,d.group(1)))
    if not prev:
        return "cleanup_old_models"

    # most-recent previous
    prev.sort(key=lambda x:x[1], reverse=True)
    latest = prev[0][0]
    try:
        art = pickle.load(open(latest,'rb'))
    except:
        return "retrain_model"

    # months since Jan 2023
    months_since = (execution_date.year - 2023)*12 + (execution_date.month - 1)
    if months_since < 12:
        return "cleanup_old_models"

    perf = art.get('performance',{})
    auc  = perf.get('oot_avg', perf.get('oot_combined_auc',0)) or 0
    cv   = perf.get('oot_cv',0) or 0

    breaches = []
    if auc>0:
        if auc<0.70: breaches.append('crit')
        if auc<0.72: breaches.append('warn')
    else:
        breaches.append('no_auc')
    if cv>0.08: breaches.append('cv')

    # PATH 4
    if len(breaches)>=2:
        return "retrain_model"

    # PATH 3
    after_grace = months_since - 12
    if after_grace>=0 and after_grace%6==0:
        return "retrain_model"

    # healthy
    return "cleanup_old_models"


with DAG(
    'complete_mlops_lifecycle_pipeline',
    default_args=default_args,
    description='Streamlined MLOps Pipeline with 4-Path Retraining Logic',
    schedule_interval='0 0 1 * *',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 1, 1),
    catchup=True,
    max_active_tasks=30,
    tags=['mlops','retraining','monitoring']
) as dag:

    start_pipeline = BashOperator(
        task_id="start_pipeline",
        bash_command=(
            'mkdir -p /opt/airflow/model_bank '
            'datamart/{bronze,silver,gold}'
        ),
    )

    check_static_data = BashOperator(
        task_id="check_static_data",
        bash_command='ls /opt/airflow/data || mkdir -p /opt/airflow/data',
    )

    # ─── PHASE 2: DEPENDENCY VALIDATION ─────────────────────────
    validate_source_data = BashOperator(
        task_id="validate_source_data",
        bash_command='echo "Validating source data..."',
    )
    dep_check_source_loans      = BashOperator(task_id="dep_check_source_loans",      bash_command='echo loans')
    dep_check_source_attributes = BashOperator(task_id="dep_check_source_attributes", bash_command='echo attributes')
    dep_check_source_financials = BashOperator(task_id="dep_check_source_financials", bash_command='echo financials')
    dep_check_source_clickstream= BashOperator(task_id="dep_check_source_clickstream",bash_command='echo clickstream')

    # ─── PHASE 3: BRONZE LAYER ───────────────────────────────────
    with TaskGroup("bronze") as bronze:
        bronze_layer       = BashOperator(task_id="bronze_layer",       bash_command='echo "Bronze: all tables"')
        bronze_loans       = BashOperator(task_id="bronze_loans",       bash_command='echo "Bronze loans"')
        bronze_attributes  = BashOperator(task_id="bronze_attributes",  bash_command='echo "Bronze attributes"')
        bronze_financials  = BashOperator(task_id="bronze_financials",  bash_command='echo "Bronze financials"')
        bronze_clickstream = BashOperator(task_id="bronze_clickstream", bash_command='echo "Bronze clickstream"')

        bronze_layer >> [
            bronze_loans, bronze_attributes,
            bronze_financials, bronze_clickstream
        ]

    # ─── PHASE 4: SILVER LAYER ───────────────────────────────────
    with TaskGroup("silver") as silver:
        silver_layer       = BashOperator(task_id="silver_layer",       bash_command='echo "Silver: all tables"')
        silver_loans       = BashOperator(task_id="silver_loans",       bash_command='echo "Silver loans"')
        silver_attributes  = BashOperator(task_id="silver_attributes",  bash_command='echo "Silver attributes"')
        silver_financials  = BashOperator(task_id="silver_financials",  bash_command='echo "Silver financials"')
        silver_clickstream = BashOperator(task_id="silver_clickstream", bash_command='echo "Silver clickstream"')

        silver_layer >> [
            silver_loans, silver_attributes,
            silver_financials, silver_clickstream
        ]

    # ─── PHASE 5: GOLD LAYER ─────────────────────────────────────
    gold_layer = BashOperator(
        task_id="gold_layer",
        bash_command='echo "Gold: feature & label stores"',
    )

    # ─── PHASE 6: INITIAL TRAIN vs INFER ─────────────────────────
    decide_pipeline_path = BranchPythonOperator(
        task_id="decide_pipeline_path",
        python_callable=decide_initial_pipeline_path,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="""
cd /opt/airflow
echo "=== RUNNING MODEL TRAINING ==="
python3 utils/train_model.py \
  --snapshotdate "{{ ds }}" \
  --config_name "credit_model_{{ ds_nodash }}" \
  --train_models xgboost \
  --mode full || test $? -eq 99
""",
        skip_exit_code=99,
    )

    best_model_selection = BashOperator(
        task_id="best_model_selection",
        bash_command="""
cd /opt/airflow
python3 utils/model_inference.py \
  --snapshotdate "{{ ds }}" \
  --modelname credit_model_{{ ds_nodash }}.pkl \
  --mode validate
""",
        trigger_rule='none_failed_or_skipped',
    )

    model_inference = BashOperator(
        task_id="model_inference",
        bash_command="""
cd /opt/airflow
echo "=== RUNNING MODEL INFERENCE ==="

# pick the latest model (by timestamped filename)
MODEL=$(ls -1t model_bank/credit_model_*.pkl 2>/dev/null | head -1)

if [ -z "$MODEL" ]; then
    echo "⚠️  No model found for inference — skipping."
    exit 0
else
    echo "✅ Using model: $MODEL"
    python3 utils/model_inference.py \
      --snapshotdate "{{ ds }}" \
      --modelname $(basename "$MODEL") \
      --mode batch_inference \
    && echo "Model inference completed successfully"
fi
""",
        trigger_rule='none_failed_or_skipped',
    )

    # ─── PHASE 9: MONITOR & RETRAIN ─────────────────────────────
    monitor_performance = BashOperator(
        task_id="monitor_performance",
        bash_command="""
cd /opt/airflow
echo "=== RUNNING MODEL MONITORING ==="

# pick the latest model (by timestamped filename)
MODEL=$(ls -1t model_bank/credit_model_*.pkl 2>/dev/null | head -1)

if [ -z "$MODEL" ]; then
    echo "⚠️  No model found for monitoring — skipping."
    exit 0
else
    echo "✅ Monitoring model: $MODEL"
    python3 utils/model_inference.py \
      --snapshotdate "{{ ds }}" \
      --modelname $(basename "$MODEL") \
      --mode monitor \
    && echo "Model monitoring completed successfully"
fi
""",
        trigger_rule='none_failed_or_skipped',
    )

    check_retraining = BranchPythonOperator(
        task_id="check_retraining_trigger",
        python_callable=check_retraining_trigger,
    )

    retrain_model = BashOperator(
        task_id="retrain_model",
        bash_command="""
cd /opt/airflow
echo "=== RUNNING MODEL RETRAINING ==="
python3 utils/train_model.py \
  --snapshotdate "{{ ds }}" \
  --config_name "credit_model_{{ ds_nodash }}_retrain" \
  --train_models xgboost \
  --mode full
""",
    )

    best_model_selection_after_retrain = BashOperator(
        task_id="best_model_selection_after_retrain",
        bash_command='echo "Compare & deploy retrained model"',
    )

    cleanup_old_models = BashOperator(
        task_id="cleanup_old_models",
        bash_command=(
            'find /opt/airflow/model_bank -maxdepth 1 -type f -name "credit_model_*.pkl" '
            '| sort -r | tail -n +4 '
            '| xargs -r -I{} mv {} {}.archived'
        ),
    )

    end_pipeline = DummyOperator(task_id="end_pipeline")

    # ─── Wiring it all together ────────────────────────────────────
    start_pipeline >> check_static_data >> validate_source_data
    validate_source_data >> [
        dep_check_source_loans, dep_check_source_attributes,
        dep_check_source_financials, dep_check_source_clickstream
    ]
    [dep_check_source_loans, dep_check_source_attributes,
     dep_check_source_financials, dep_check_source_clickstream] >> bronze >> silver >> gold_layer
    gold_layer >> decide_pipeline_path

    # when no model yet:
    decide_pipeline_path >> train_model >> best_model_selection >> model_inference
    # when model(s) exist:
    decide_pipeline_path >> model_inference

    model_inference >> monitor_performance >> check_retraining
    check_retraining >> retrain_model  >> best_model_selection_after_retrain >> cleanup_old_models >> end_pipeline
    check_retraining >> cleanup_old_models >> end_pipeline
