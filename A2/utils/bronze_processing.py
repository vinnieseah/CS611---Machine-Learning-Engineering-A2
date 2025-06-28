from pyspark.sql.functions import col 
import os

def ingest_bronze_tables(spark, table_filter=None):
    """
    Ingest CSV data sources into bronze layer parquet tables.
    
    Args:
        spark: Spark session
        table_filter: Optional filter to process only specific tables (e.g., "loans", "clickstream")
    """
    sources = {
        "clickstream": "data/feature_clickstream.csv",
        "attributes": "data/features_attributes.csv", 
        "financials": "data/features_financials.csv",
        "loans": "data/lms_loan_daily.csv"
    }
    
    # Filter sources if table_filter is provided
    if table_filter:
        if table_filter not in sources:
            print(f"WARNING: Table filter '{table_filter}' not found in available sources: {list(sources.keys())}")
            print("Processing all available sources instead...")
        else:
            sources = {table_filter: sources[table_filter]}
            print(f"Processing only: {table_filter}")
    
    # Ensure bronze directory exists
    os.makedirs("datamart/bronze", exist_ok=True)
    
    for name, path in sources.items():
        try:
            print(f"Ingesting {name} from {path}")
            
            # Check if source file exists
            if not os.path.exists(path):
                print(f"ERROR: Source file not found: {path}")
                continue
                
            df = spark.read.csv(path, header=True, inferSchema=True)
            row_count = df.count()
            print(f"  Loaded {row_count} rows from {path}")
            
            # Ensure bronze subdirectory exists
            bronze_path = f"datamart/bronze/{name}"
            os.makedirs(bronze_path, exist_ok=True)
            
            # Partition time series sources by snapshot_date
            if "snapshot_date" in df.columns and name in ["clickstream", "financials"]:
                print(f"  Partitioning {name} by snapshot_date")
                df.write.partitionBy("snapshot_date").mode("overwrite").parquet(bronze_path)
            else:
                print(f"  Writing {name} without partitioning")
                df.write.mode("overwrite").parquet(bronze_path)

            print(f"Saved to bronze/{name}")
            
        except Exception as e:
            print(f"ERROR processing {name}: {str(e)}")
            raise e
    
    print("Bronze ingestion completed successfully")
    return True 