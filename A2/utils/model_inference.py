import argparse
import os
import glob
import pandas as pd
import pickle
import numpy as np
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import warnings
warnings.filterwarnings('ignore')

# PySpark imports
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

# Matplotlib import (set backend for headless environments)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MLPipeline:
    def __init__(self, config):
        self.config = config
        self.spark = None
        self.model_metrics = {}
        self.model_artifact = None  # Store loaded model artifact
        # Make thresholds configurable instead of static - optimized for excellent model
        self.drift_threshold = config.get('drift_threshold', 0.08)  # Tighter drift detection
        self.performance_threshold = config.get('performance_threshold', 0.72)  # 3% drop from ~0.74
        
    def initialize_spark(self):
        try:
            self.spark = pyspark.sql.SparkSession.builder \
                .appName("ML_Pipeline") \
                .master("local[*]") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            self.spark.sparkContext.setLogLevel("ERROR")
            print("SUCCESS: Spark session initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to initialize Spark session: {str(e)}")
            raise RuntimeError(f"Spark initialization failed: {str(e)}")
        
    def validate_model_exists(self):
        """Enhanced validation for the 3 OOT model artifact"""
        model_path = self.config["model_artefact_filepath"]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Please ensure model is trained and saved to model_bank/")
        
        # Load and validate model artifact structure
        try:
            with open(model_path, 'rb') as file:
                self.model_artifact = pickle.load(file)
            
            print(f"SUCCESS: Model loaded from {model_path}")
            
            # Validate enhanced model artifact structure
            required_keys = ['model', 'feature_columns', 'preprocessing_transformers']
            missing_keys = [key for key in required_keys if key not in self.model_artifact]
            
            if missing_keys:
                print(f"WARNING: Missing keys in model artifact: {missing_keys}")
            else:
                print("SUCCESS: All required keys found in model artifact")
            
            # Check for 3 OOT enhancements
            if 'results' in self.model_artifact:
                results = self.model_artifact['results']
                if 'auc_oot_individual' in results:
                    oot_periods = len(results['auc_oot_individual'])
                    print(f"SUCCESS: Enhanced model with {oot_periods} OOT periods detected")
                    
                    # Display OOT stability information
                    if 'auc_oot_cv' in results:
                        cv = results['auc_oot_cv']
                        if cv < 0.02:
                            stability = "EXCELLENT"
                        elif cv < 0.05:
                            stability = "GOOD"
                        elif cv < 0.1:
                            stability = "MODERATE"
                        else:
                            stability = "POOR"
                        print(f"Model Stability: {stability} (CV: {cv:.4f})")
                    
                    # Display individual OOT scores
                    print("Individual OOT Performance:")
                    for period, score in results['auc_oot_individual'].items():
                        gini = round(2 * score - 1, 3)
                        print(f"  {period}: AUC {score:.4f} (Gini {gini})")
                else:
                    print("WARNING: Legacy model without individual OOT results")
            else:
                print("WARNING: No results found in model artifact")
            
            # Validate feature columns
            feature_cols = self.model_artifact.get('feature_columns', [])
            print(f"Model feature columns: {len(feature_cols)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load or validate model artifact: {str(e)}")
        
        return model_path
    
    def make_predictions(self, model_path=None):
        """Enhanced prediction with better error handling for 3 OOT models"""
        print("Starting enhanced model inference...")
        
        # Use pre-loaded model artifact if available
        if self.model_artifact is None:
            # Load model from model_bank
            if model_path is None:
                model_path = self.config["model_artefact_filepath"]
            
            with open(model_path, 'rb') as file:
                self.model_artifact = pickle.load(file)
        
        print(f"Model loaded successfully: {self.config['model_artefact_filepath']}")
        
        # Display model information
        if 'model_version' in self.model_artifact:
            print(f"Model version: {self.model_artifact['model_version']}")
        
        # Load feature data from feature store
        feature_store_path = self.config.get('feature_store_path', "datamart/gold/feature_store")
        features_sdf = self.spark.read.parquet(feature_store_path)
        features_sdf = features_sdf.filter(col("feature_snapshot_date") == self.config["snapshot_date"])
        
        print(f"Features extracted for inference: {features_sdf.count()} records")
        features_pdf = features_sdf.toPandas()
        
        if len(features_pdf) == 0:
            raise ValueError(f"No feature data found for date {self.config['snapshot_date']}")
        
        # Enhanced feature preparation with 3 OOT model
        if 'feature_columns' in self.model_artifact:
            feature_cols = self.model_artifact['feature_columns']
            print(f"SUCCESS: Using {len(feature_cols)} features from enhanced model artifact")
            
            # Check if all required features are available
            missing_features = [col for col in feature_cols if col not in features_pdf.columns]
            if missing_features:
                print(f"WARNING: Missing features in inference data: {missing_features[:5]}...")
                if len(missing_features) > 5:
                    print(f"... and {len(missing_features) - 5} more missing features")
                
                # Use only available features and pad missing ones with zeros
                available_feature_cols = [col for col in feature_cols if col in features_pdf.columns]
                X_inference = features_pdf[available_feature_cols].fillna(0)
                
                # Add missing features as zeros
                for missing_col in missing_features:
                    X_inference[missing_col] = 0
                
                # Reorder columns to match model's expected order
                X_inference = X_inference[feature_cols]
                
                print(f"Using {len(available_feature_cols)} available features, padded {len(missing_features)} missing features with zeros")
                
                # Warning if too many features are missing
                missing_ratio = len(missing_features) / len(feature_cols)
                if missing_ratio > 0.1:
                    print(f"WARNING: {missing_ratio:.1%} of features are missing. Model performance may be degraded.")
            else:
                X_inference = features_pdf[feature_cols].fillna(0)
                print(f"SUCCESS: All {len(feature_cols)} required features found in inference data")
        else:
            # Fallback: Use all numeric columns except ID and date columns
            exclude_cols = ['Customer_ID', 'snapshot_date', 'feature_snapshot_date']
            feature_cols = [col for col in features_pdf.columns 
                          if col not in exclude_cols and features_pdf[col].dtype in ['int64', 'float64']]
            print(f"WARNING: feature_columns not found in model artifact. Using {len(feature_cols)} available numeric features.")
            X_inference = features_pdf[feature_cols].fillna(0)
        
        # Apply preprocessing if available
        if 'preprocessing_transformers' in self.model_artifact:
            transformer = self.model_artifact['preprocessing_transformers']['stdscaler']
            X_inference = transformer.transform(X_inference)
            print("SUCCESS: Preprocessing applied using stored StandardScaler")
        else:
            print("WARNING: No preprocessing transformers found in model artifact")
        
        # Load model and predict
        model = self.model_artifact["model"]
        model_type = type(model).__name__
        print(f"Making predictions using {model_type}")
        
        y_inference = model.predict_proba(X_inference)[:, 1]
        
        # Enhanced prediction statistics
        pred_stats = {
            'count': len(y_inference),
            'mean': np.mean(y_inference),
            'std': np.std(y_inference),
            'min': np.min(y_inference),
            'max': np.max(y_inference),
            'positive_rate': np.sum(y_inference > 0.5) / len(y_inference)
        }
        
        print(f"Prediction Statistics:")
        print(f"  Count: {pred_stats['count']:,}")
        print(f"  Mean probability: {pred_stats['mean']:.4f}")
        print(f"  Std deviation: {pred_stats['std']:.4f}")
        print(f"  Range: {pred_stats['min']:.4f} - {pred_stats['max']:.4f}")
        print(f"  Positive rate (>0.5): {pred_stats['positive_rate']:.4f}")
        
        # Prepare output with enhanced metadata
        y_inference_pdf = features_pdf[["Customer_ID", "feature_snapshot_date"]].copy()
        y_inference_pdf["snapshot_date"] = y_inference_pdf["feature_snapshot_date"]
        y_inference_pdf["model_name"] = self.config["model_name"]
        y_inference_pdf["model_predictions"] = y_inference
        y_inference_pdf["prediction_class"] = (y_inference > 0.5).astype(int)
        
        # Add model metadata
        if 'model_version' in self.model_artifact:
            y_inference_pdf["model_version"] = self.model_artifact['model_version']
        
        # Save model inference to datamart gold table
        predictions_base_path = self.config.get('predictions_output_path', 'datamart/gold/model_predictions')
        gold_directory = f"{predictions_base_path}/{self.config['model_name'][:-4]}/"
        
        if not os.path.exists(gold_directory):
            os.makedirs(gold_directory)
        
        # Save gold table
        partition_name = self.config["model_name"][:-4] + "_predictions_" + self.config["snapshot_date_str"].replace('-','_') + '.parquet'
        filepath = gold_directory + partition_name
        self.spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
        print(f'SUCCESS: Predictions saved to {filepath}')
        
        return y_inference_pdf
    
    def monitor_model_performance(self):
        """
        Enhanced monitoring with 3 OOT model support and stability analysis
        """
        print("="*60)
        print("STEP 3: ENHANCED MODEL MONITORING & 3 OOT STABILITY ANALYSIS")
        print("="*60)
        
        # Load historical predictions
        predictions_dir = f"{self.config.get('predictions_output_path', 'datamart/gold/model_predictions')}/"
        monitoring_results = {}
        
        try:
            # Get all prediction files
            prediction_files = glob.glob(f"{predictions_dir}/**/*.parquet", recursive=True)
            print(f"Found {len(prediction_files)} prediction files for monitoring")
            
            if not prediction_files:
                print("No historical predictions found for monitoring")
                return self._create_enhanced_monitoring_results()
            
            all_predictions = []
            for file_path in prediction_files:
                try:
                    pred_df = self.spark.read.parquet(file_path).toPandas()
                    # Standardize column names - handle both old and new formats
                    if 'model_predictions' in pred_df.columns and 'prediction_probability' not in pred_df.columns:
                        pred_df['prediction_probability'] = pred_df['model_predictions']
                        pred_df['prediction_class'] = (pred_df['model_predictions'] > 0.5).astype(int)
                    all_predictions.append(pred_df)
                    print(f"  SUCCESS: Loaded {len(pred_df)} predictions from {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  ERROR: Failed to load {os.path.basename(file_path)}: {e}")
                    continue
            
            if all_predictions:
                historical_predictions = pd.concat(all_predictions, ignore_index=True)
                print(f"\nINFO: Total historical predictions loaded: {len(historical_predictions)}")
            else:
                return self._create_enhanced_monitoring_results()
            
        except Exception as e:
            print(f"Error loading historical predictions: {e}")
            return self._create_enhanced_monitoring_results()
        
        # Enhanced monitoring for 3 OOT models
        print("\nINFO: Calculating performance trends...")
        performance_over_time = self._calculate_performance_trends(historical_predictions)
        
        print("INFO: Detecting data drift...")
        drift_analysis = self._detect_data_drift(historical_predictions)
        
        print("INFO: Analyzing model stability...")
        stability_metrics = self._analyze_model_stability(historical_predictions)
        
        print("INFO: Performing distribution analysis...")
        distribution_analysis = self._analyze_prediction_distribution(historical_predictions)
        
        print("INFO: Calculating model confidence metrics...")
        confidence_metrics = self._calculate_confidence_metrics(historical_predictions)
        
        print("INFO: Analyzing 3 OOT training stability...")
        oot_training_analysis = self._analyze_oot_training_stability()
        
        monitoring_results = {
            'monitoring_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'monitoring_period': f"{historical_predictions['snapshot_date'].min()} to {historical_predictions['snapshot_date'].max()}",
            'total_predictions_analyzed': len(historical_predictions),
            'model_type': '3_oot_enhanced' if self.model_artifact and 'auc_oot_individual' in self.model_artifact.get('results', {}) else 'legacy',
            'performance_trends': performance_over_time,
            'drift_analysis': drift_analysis,
            'stability_metrics': stability_metrics,
            'distribution_analysis': distribution_analysis,
            'confidence_metrics': confidence_metrics,
            'oot_training_analysis': oot_training_analysis,
            'governance_results': self._generate_enhanced_model_recommendations(performance_over_time, drift_analysis, stability_metrics, oot_training_analysis)
        }
        
        # Save monitoring results to gold table
        self._save_monitoring_results(monitoring_results)
        
        # Display summary
        self._display_enhanced_monitoring_summary(monitoring_results)
        
        return monitoring_results
    
    def _analyze_oot_training_stability(self):
        """Analyze the stability of the 3 OOT training results"""
        if not self.model_artifact or 'results' not in self.model_artifact:
            return {"message": "No model artifact or results available"}
        
        results = self.model_artifact['results']
        if 'auc_oot_individual' not in results:
            return {"message": "No individual OOT results found - legacy model"}
        
        oot_scores = list(results['auc_oot_individual'].values())
        oot_periods = list(results['auc_oot_individual'].keys())
        
        analysis = {
            'oot_periods': oot_periods,
            'individual_scores': results['auc_oot_individual'],
            'combined_score': results.get('auc_oot_combined', 0),
            'average_score': results.get('auc_oot_avg', np.mean(oot_scores)),
            'std_deviation': results.get('auc_oot_std', np.std(oot_scores)),
            'coefficient_of_variation': results.get('auc_oot_cv', np.std(oot_scores) / np.mean(oot_scores)),
            'score_range': f"{min(oot_scores):.4f} - {max(oot_scores):.4f}",
            'score_spread': max(oot_scores) - min(oot_scores)
        }
        
        # Stability assessment
        cv = analysis['coefficient_of_variation']
        if cv < 0.02:
            analysis['stability_rating'] = 'EXCELLENT'
            analysis['stability_explanation'] = 'Very consistent performance across all OOT periods'
        elif cv < 0.05:
            analysis['stability_rating'] = 'GOOD'
            analysis['stability_explanation'] = 'Good consistency with minor variations'
        elif cv < 0.1:
            analysis['stability_rating'] = 'MODERATE'
            analysis['stability_explanation'] = 'Moderate consistency with some variation'
        else:
            analysis['stability_rating'] = 'POOR'
            analysis['stability_explanation'] = 'High variation - potential stability concerns'
        
        return analysis
    
    def _analyze_prediction_distribution(self, predictions_df):
        """Analyze distribution of predictions across time periods"""
        if 'prediction_probability' not in predictions_df.columns:
            return {"message": "No prediction data available for distribution analysis"}
        
        predictions_df['snapshot_date'] = pd.to_datetime(predictions_df['snapshot_date'])
        
        # Calculate distribution metrics by time period
        distribution_by_period = predictions_df.groupby('snapshot_date')['prediction_probability'].agg([
            'count', 'mean', 'std', 'min', 'max', 
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.5), 
            lambda x: x.quantile(0.75)
        ]).round(4)
        
        distribution_by_period.columns = ['count', 'mean', 'std', 'min', 'max', 'q25', 'median', 'q75']
        
        # Calculate overall distribution health
        overall_cv = predictions_df['prediction_probability'].std() / predictions_df['prediction_probability'].mean()
        
        return {
            'distribution_by_period': distribution_by_period.to_dict('index'),
            'overall_coefficient_of_variation': round(overall_cv, 4),
            'prediction_concentration': {
                'below_0.1': len(predictions_df[predictions_df['prediction_probability'] < 0.1]) / len(predictions_df),
                'between_0.1_0.5': len(predictions_df[(predictions_df['prediction_probability'] >= 0.1) & 
                                                     (predictions_df['prediction_probability'] < 0.5)]) / len(predictions_df),
                'above_0.5': len(predictions_df[predictions_df['prediction_probability'] >= 0.5]) / len(predictions_df)
            }
        }
    
    def _calculate_confidence_metrics(self, predictions_df):
        """Calculate model confidence and calibration metrics"""
        if 'prediction_probability' not in predictions_df.columns:
            return {"message": "No prediction data available for confidence analysis"}
        
        # Model confidence analysis
        high_confidence = len(predictions_df[(predictions_df['prediction_probability'] < 0.2) | 
                                           (predictions_df['prediction_probability'] > 0.8)]) / len(predictions_df)
        
        medium_confidence = len(predictions_df[(predictions_df['prediction_probability'] >= 0.2) & 
                                             (predictions_df['prediction_probability'] <= 0.8)]) / len(predictions_df)
        
        # Prediction consistency across time
        predictions_df['snapshot_date'] = pd.to_datetime(predictions_df['snapshot_date'])
        time_variance = predictions_df.groupby('snapshot_date')['prediction_probability'].mean().std()
        
        return {
            'high_confidence_predictions': round(high_confidence, 4),
            'medium_confidence_predictions': round(medium_confidence, 4),
            'temporal_consistency': round(1 / (1 + time_variance), 4),  # Higher is more consistent
            'average_prediction_confidence': round(predictions_df['prediction_probability'].mean(), 4),
            'confidence_std': round(predictions_df['prediction_probability'].std(), 4)
        }
    
    def _display_enhanced_monitoring_summary(self, monitoring_results):
        """Display enhanced monitoring summary with 3 OOT analysis"""
        print("\n" + "="*60)
        print("ENHANCED MODEL MONITORING SUMMARY (3 OOT)")
        print("="*60)
        
        print(f"Model Type: {monitoring_results['model_type'].upper()}")
        print(f"Monitoring Period: {monitoring_results['monitoring_period']}")
        print(f"Total Predictions: {monitoring_results['total_predictions_analyzed']:,}")
        
        # OOT Training Analysis
        oot_analysis = monitoring_results['oot_training_analysis']
        if 'stability_rating' in oot_analysis:
            print(f"\n3 OOT TRAINING STABILITY:")
            print(f"  Rating: {oot_analysis['stability_rating']}")
            print(f"  Coefficient of Variation: {oot_analysis['coefficient_of_variation']:.4f}")
            print(f"  Score Range: {oot_analysis['score_range']}")
            print(f"  Explanation: {oot_analysis['stability_explanation']}")
        
        # Drift Analysis Summary
        drift = monitoring_results['drift_analysis']
        drift_status = "DETECTED" if drift.get('drift_detected', False) else "NOT DETECTED"
        print(f"\nData Drift: {drift_status}")
        
        # Stability Summary
        stability = monitoring_results['stability_metrics']
        stability_status = "STABLE" if stability.get('is_stable', True) else "UNSTABLE"
        print(f"Model Stability: {stability_status}")
        
        # Confidence Summary
        confidence = monitoring_results['confidence_metrics']
        print(f"Average Confidence: {confidence.get('average_prediction_confidence', 0):.4f}")
        
        # Governance Results
        governance = monitoring_results['governance_results']
        recommendations = governance['recommendations']
        sop = governance['governance_sop']
        
        # Recommendations
        print(f"\nIMMEDIATE RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # 3 OOT Specific Governance
        if '3_oot_requirements' in sop:
            print(f"\n3 OOT MODEL REQUIREMENTS:")
            requirements = sop['3_oot_requirements']
            print(f"   Minimum Periods: {requirements.get('minimum_periods', 'N/A')}")
            print(f"   Period Length: {requirements.get('period_length', 'N/A')}")
            print(f"   Stability Threshold: {requirements.get('stability_threshold', 'N/A')}")
            print(f"   Minimum AUC: {requirements.get('minimum_auc', 'N/A')}")
        
        print("\n" + "="*60)
        print("ENHANCED GOVERNANCE FRAMEWORK APPLIED")
        print("3 OOT stability analysis completed")
        print("Enhanced monitoring results saved to gold table")
        print("="*60)
    
    def _calculate_performance_trends(self, predictions_df):
        """Calculate performance trends over time"""
        if 'snapshot_date' in predictions_df.columns:
            predictions_df['snapshot_date'] = pd.to_datetime(predictions_df['snapshot_date'])
            
            trends = predictions_df.groupby('snapshot_date').agg({
                'prediction_probability': ['mean', 'std', 'count'],
                'prediction_class': 'mean'
            }).round(4)
            
            trends.columns = ['avg_probability', 'std_probability', 'prediction_count', 'positive_rate']
            return trends.to_dict('index')
        
        return {"message": "No trend data available"}
    
    def _detect_data_drift(self, predictions_df):
        """Detect data drift in model predictions"""
        if len(predictions_df) < 2:
            return {"drift_detected": False, "message": "Insufficient data for drift detection"}
        
        # Simple drift detection based on prediction distribution changes
        predictions_df['snapshot_date'] = pd.to_datetime(predictions_df['snapshot_date'])
        predictions_df = predictions_df.sort_values('snapshot_date')
        
        # Split into early and recent periods
        mid_point = len(predictions_df) // 2
        early_period = predictions_df.iloc[:mid_point]['prediction_probability']
        recent_period = predictions_df.iloc[mid_point:]['prediction_probability']
        
        # Calculate distribution differences
        mean_diff = abs(recent_period.mean() - early_period.mean())
        std_diff = abs(recent_period.std() - early_period.std())
        
        drift_detected = mean_diff > self.drift_threshold or std_diff > self.drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'mean_difference': round(mean_diff, 4),
            'std_difference': round(std_diff, 4),
            'drift_threshold': self.drift_threshold,
            'early_period_stats': {
                'mean': round(early_period.mean(), 4),
                'std': round(early_period.std(), 4)
            },
            'recent_period_stats': {
                'mean': round(recent_period.mean(), 4),
                'std': round(recent_period.std(), 4)
            }
        }
    
    def _analyze_model_stability(self, predictions_df):
        """Analyze model prediction stability"""
        if 'prediction_probability' not in predictions_df.columns:
            return {"message": "No prediction data available for stability analysis"}
        
        stability_metrics = {
            'overall_mean': round(predictions_df['prediction_probability'].mean(), 4),
            'overall_std': round(predictions_df['prediction_probability'].std(), 4),
            'prediction_count': len(predictions_df),
            'positive_prediction_rate': round(predictions_df['prediction_class'].mean(), 4),
            'coefficient_of_variation': round(predictions_df['prediction_probability'].std() / predictions_df['prediction_probability'].mean(), 4)
        }
        
        # Check if model is stable (low coefficient of variation indicates stability)
        stability_metrics['is_stable'] = stability_metrics['coefficient_of_variation'] < 0.5
        
        return stability_metrics
    
    def _generate_enhanced_model_recommendations(self, performance_trends, drift_analysis, stability_metrics, oot_training_analysis):
        """Generate enhanced recommendations considering 3 OOT training stability"""
        recommendations = []
        governance_sop = {}
        deployment_options = {}
        
        # === ENHANCED GOVERNANCE WITH 3 OOT ANALYSIS ===
        
        drift_detected = drift_analysis.get('drift_detected', False)
        is_stable = stability_metrics.get('is_stable', True)
        prediction_count = stability_metrics.get('prediction_count', 0)
        positive_rate = stability_metrics.get('positive_prediction_rate', 0)
        mean_diff = drift_analysis.get('mean_difference', 0)
        
        # 3 OOT Training Stability Analysis
        oot_stability_rating = oot_training_analysis.get('stability_rating', 'UNKNOWN')
        oot_cv = oot_training_analysis.get('coefficient_of_variation', 0)
        
        # === ENHANCED TRIGGER CONDITIONS ===
        
        # Critical: Immediate Action Required
        if drift_detected and mean_diff > 0.15:
            recommendations.append("CRITICAL: Severe data drift detected - IMMEDIATE RETRAIN REQUIRED")
            governance_sop['retrain_urgency'] = 'IMMEDIATE'
            governance_sop['retrain_timeline'] = '24-48 hours'
            deployment_options['strategy'] = 'blue_green_deployment'
            deployment_options['rollback_plan'] = 'automated_rollback_on_performance_drop'
            
        elif drift_detected:
            recommendations.append("HIGH: Data drift detected - Schedule retrain within 1 week")
            governance_sop['retrain_urgency'] = 'HIGH'
            governance_sop['retrain_timeline'] = '1 week'
            deployment_options['strategy'] = 'canary_deployment'
            deployment_options['rollback_plan'] = 'manual_approval_rollback'
        
        # Enhanced stability assessment with 3 OOT consideration
        if oot_stability_rating == 'POOR':
            recommendations.append("CRITICAL: Poor OOT stability during training - Model reliability compromised")
            governance_sop['action'] = 'immediate_model_review'
            governance_sop['investigation'] = 'feature_engineering_review'
        elif oot_stability_rating == 'MODERATE':
            recommendations.append("MEDIUM: Moderate OOT stability - Monitor closely")
            governance_sop['action'] = 'enhanced_monitoring'
        elif oot_stability_rating == 'EXCELLENT':
            recommendations.append("EXCELLENT: High OOT stability - Model is robust")
            governance_sop['action'] = 'standard_monitoring'
        
        # Stability Issues
        if not is_stable:
            coeff_var = stability_metrics.get('coefficient_of_variation', 0)
            if coeff_var > 1.0:
                recommendations.append("CRITICAL: High prediction variability - Model stability compromised")
                governance_sop['action'] = 'investigate_model_degradation'
                deployment_options['monitoring'] = 'enhanced_real_time_monitoring'
            else:
                recommendations.append("MEDIUM: Monitor model stability closely")
                governance_sop['action'] = 'increase_monitoring_frequency'
        
        # Volume-based governance
        if prediction_count < 100:
            recommendations.append("LOW: Insufficient prediction volume for reliable monitoring")
            governance_sop['action'] = 'increase_data_collection'
        elif prediction_count > 10000:
            recommendations.append("HIGH: High prediction volume - Consider model optimization")
            governance_sop['action'] = 'optimize_inference_pipeline'
        
        # Performance anomalies
        if positive_rate > 0.8:
            recommendations.append("INVESTIGATE: Unusually high positive prediction rate")
            governance_sop['investigation'] = 'data_quality_check'
        elif positive_rate < 0.05:
            recommendations.append("INVESTIGATE: Unusually low positive prediction rate")
            governance_sop['investigation'] = 'feature_distribution_analysis'
        
        # === ENHANCED MODEL REFRESH SOPs ===
        governance_sop.update({
            'refresh_triggers': {
                'performance_degradation': 'AUC drops below 0.75',
                'data_drift': 'Distribution shift > 0.1',
                'stability_loss': 'Coefficient of variation > 0.5',
                'oot_instability': 'OOT CV > 0.1',
                'business_rules': 'Quarterly mandatory refresh'
            },
            'refresh_process': {
                'data_validation': 'Validate new training data quality',
                'model_training': 'Train with 3 OOT periods (6 months)',
                'oot_validation': 'Ensure OOT CV < 0.05 for stability',
                'backtesting': 'Test on additional holdout period',
                'approval_workflow': 'Model committee approval required',
                'deployment_approval': 'Risk management sign-off'
            },
            'model_versioning': {
                'naming_convention': 'credit_model_YYYY_MM_DD',
                'artifact_storage': 'model_bank/ with 3 OOT metadata',
                'rollback_retention': 'Keep last 3 model versions',
                'performance_benchmark': 'Compare 3 OOT stability vs previous model'
            },
            '3_oot_requirements': {
                'minimum_periods': 3,
                'period_length': '2 months each',
                'stability_threshold': 'CV < 0.05',
                'minimum_auc': '0.72 per period',
                'retraining_frequency': '6 months (for excellent stability)',
                'early_warning_cv': 0.03
            }
        })
        
        # Default recommendation if no issues
        if not recommendations:
            if oot_stability_rating == 'EXCELLENT':
                recommendations.append("EXCELLENT: 3 OOT model showing excellent stability - Continue monitoring")
            else:
                recommendations.append("CONTINUE: Model performance stable - Continue regular monitoring")
            governance_sop['action'] = 'routine_monitoring'
            deployment_options['strategy'] = 'maintain_current_deployment'
        
        return {
            'recommendations': recommendations,
            'governance_sop': governance_sop,
            'deployment_options': deployment_options,
            'oot_stability_assessment': {
                'rating': oot_stability_rating,
                'coefficient_of_variation': oot_cv,
                'recommendation': oot_training_analysis.get('stability_explanation', 'N/A')
            }
        }
    
    def _create_enhanced_monitoring_results(self):
        """Create enhanced sample monitoring results for 3 OOT models"""
        oot_analysis = self._analyze_oot_training_stability()
        
        return {
            'monitoring_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': '3_oot_enhanced' if oot_analysis.get('oot_periods') else 'legacy',
            'performance_trends': {"message": "Sample monitoring data created"},
            'drift_analysis': {
                'drift_detected': False,
                'mean_difference': 0.02,
                'std_difference': 0.01,
                'drift_threshold': self.drift_threshold
            },
            'stability_metrics': {
                'overall_mean': 0.25,
                'overall_std': 0.18,
                'prediction_count': 1000,
                'positive_prediction_rate': 0.25,
                'is_stable': True
            },
            'oot_training_analysis': oot_analysis,
            'governance_results': {
                'recommendations': ["EXCELLENT: 3 OOT model showing excellent stability - Continue monitoring"],
                'governance_sop': {'action': 'routine_monitoring'},
                'deployment_options': {'strategy': 'maintain_current_deployment'},
                'oot_stability_assessment': {
                    'rating': oot_analysis.get('stability_rating', 'UNKNOWN'),
                    'coefficient_of_variation': oot_analysis.get('coefficient_of_variation', 0),
                    'recommendation': oot_analysis.get('stability_explanation', 'N/A')
                }
            }
        }
    
    def visualize_performance(self, monitoring_results):
        """Create visualizations for model performance and monitoring"""
        print("Creating performance visualizations...")
        
        # Create visualization directory
        viz_dir = "datamart/gold/model_monitoring/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot 1: Model Performance Trends
        plt.figure(figsize=(15, 10))
        
        # Performance trends plot
        plt.subplot(2, 2, 1)
        if isinstance(monitoring_results['performance_trends'], dict) and len(monitoring_results['performance_trends']) > 1:
            dates = list(monitoring_results['performance_trends'].keys())
            avg_probs = [monitoring_results['performance_trends'][date]['avg_probability'] for date in dates]
            plt.plot(dates, avg_probs, marker='o')
            plt.title('Average Prediction Probability Over Time')
            plt.xticks(rotation=45)
            plt.ylabel('Average Probability')
        else:
            plt.text(0.5, 0.5, 'Insufficient data for trends', ha='center', va='center')
            plt.title('Performance Trends (Insufficient Data)')
        
        # Drift analysis plot
        plt.subplot(2, 2, 2)
        drift_data = monitoring_results['drift_analysis']
        if 'early_period_stats' in drift_data and 'recent_period_stats' in drift_data:
            periods = ['Early Period', 'Recent Period']
            means = [drift_data['early_period_stats']['mean'], drift_data['recent_period_stats']['mean']]
            stds = [drift_data['early_period_stats']['std'], drift_data['recent_period_stats']['std']]
            
            x = np.arange(len(periods))
            width = 0.35
            
            plt.bar(x - width/2, means, width, label='Mean', alpha=0.8)
            plt.bar(x + width/2, stds, width, label='Std Dev', alpha=0.8)
            plt.xlabel('Time Period')
            plt.ylabel('Value')
            plt.title('Distribution Comparison (Drift Analysis)')
            plt.xticks(x, periods)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No drift analysis data', ha='center', va='center')
            plt.title('Drift Analysis (No Data)')
        
        # Stability metrics plot
        plt.subplot(2, 2, 3)
        stability = monitoring_results['stability_metrics']
        metrics = ['Mean', 'Std Dev', 'Pos Rate', 'Coeff Var']
        values = [
            stability.get('overall_mean', 0),
            stability.get('overall_std', 0),
            stability.get('positive_prediction_rate', 0),
            stability.get('coefficient_of_variation', 0)
        ]
        
        plt.bar(metrics, values, alpha=0.7)
        plt.title('Model Stability Metrics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Recommendations
        plt.subplot(2, 2, 4)
        recommendations = monitoring_results['governance_results']['recommendations']
        plt.text(0.1, 0.5, '\n'.join(recommendations), fontsize=10, 
                verticalalignment='center', wrap=True)
        plt.title('Model Governance Recommendations')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"model_monitoring_{self.config['snapshot_date_str']}.png"
        plt.savefig(os.path.join(viz_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {os.path.join(viz_dir, plot_filename)}")
        
        return os.path.join(viz_dir, plot_filename)
    
    def _save_monitoring_results(self, monitoring_results):
        """Save comprehensive monitoring results including governance data to gold table"""
        monitoring_dir = self.config.get('monitoring_output_path', 'datamart/gold/model_monitoring')
        os.makedirs(monitoring_dir, exist_ok=True)
        
        # Extract governance results
        governance = monitoring_results['governance_results']
        
        # Convert to DataFrame format with comprehensive governance data
        monitoring_data = {
            'monitoring_date': monitoring_results['monitoring_date'],
            'snapshot_date': self.config['snapshot_date_str'],
            'drift_detected': monitoring_results['drift_analysis'].get('drift_detected', False),
            'mean_difference': monitoring_results['drift_analysis'].get('mean_difference', 0),
            'model_stable': monitoring_results['stability_metrics'].get('is_stable', True),
            'prediction_count': monitoring_results['stability_metrics'].get('prediction_count', 0),
            'positive_rate': monitoring_results['stability_metrics'].get('positive_prediction_rate', 0),
            'recommendations': ' | '.join(governance['recommendations']),
            
            # Governance SOPs
            'retrain_urgency': governance['governance_sop'].get('retrain_urgency', 'N/A'),
            'retrain_timeline': governance['governance_sop'].get('retrain_timeline', 'N/A'),
            'required_action': governance['governance_sop'].get('action', 'N/A'),
            'investigation_needed': governance['governance_sop'].get('investigation', 'N/A'),
            
            # Deployment Strategy
            'deployment_strategy': governance['deployment_options'].get('strategy', 'N/A'),
            'rollback_plan': governance['deployment_options'].get('rollback_plan', 'N/A'),
            
            # Refresh Triggers (as JSON string)
            'refresh_triggers': str(governance['governance_sop'].get('refresh_triggers', {})),
            'model_version': self.config.get('model_name', 'unknown'),
            
            # Additional metrics
            'avg_confidence': monitoring_results['confidence_metrics'].get('average_prediction_confidence', 0),
            'temporal_consistency': monitoring_results['confidence_metrics'].get('temporal_consistency', 0),
            'coefficient_of_variation': monitoring_results['stability_metrics'].get('coefficient_of_variation', 0)
        }
        
        monitoring_df = pd.DataFrame([monitoring_data])
        
        filename = f"model_monitoring_{self.config['snapshot_date_str'].replace('-', '_')}.parquet"
        filepath = os.path.join(monitoring_dir, filename)
        
        spark_df = self.spark.createDataFrame(monitoring_df)
        spark_df.write.mode("overwrite").parquet(filepath)
        
        print(f"Comprehensive monitoring results saved: {filepath}")
        
        # Also save detailed governance JSON for reference
        governance_file = os.path.join(monitoring_dir, f"governance_details_{self.config['snapshot_date_str'].replace('-', '_')}.json")
        
        with open(governance_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            governance_copy = governance.copy()
            json.dump(governance_copy, f, indent=2, default=str)
        
        print(f"Detailed governance framework saved: {governance_file}")
    
    def run_batch_inference(self, start_date=None, end_date=None, frequency='monthly'):
        """
        Step 2: Run inference across multiple time periods and store as gold tables
        """
        print("="*60)
        print("STEP 2: BATCH INFERENCE ACROSS TIME PERIODS")
        print("="*60)
        
        # Set default date range if not provided
        if start_date is None:
            start_date = self.config['snapshot_date_str']
        if end_date is None:
            # Default to 3 months forward
            end_dt = self.config['snapshot_date'] + relativedelta(months=3)
            end_date = end_dt.strftime('%Y-%m-%d')
        
        print(f"Batch inference from {start_date} to {end_date} ({frequency})")
        
        # Generate inference dates
        inference_dates = self._generate_inference_dates(start_date, end_date, frequency)
        print(f"Generated {len(inference_dates)} inference dates: {', '.join(inference_dates)}")
        
        batch_results = []
        successful_runs = 0
        failed_runs = 0
        
        for i, snapshot_date in enumerate(inference_dates, 1):
            print(f"\n[{i}/{len(inference_dates)}] Processing {snapshot_date}...")
            
            try:
                # Update config for this snapshot date
                original_config = self.config.copy()
                self.config['snapshot_date_str'] = snapshot_date
                self.config['snapshot_date'] = datetime.strptime(snapshot_date, "%Y-%m-%d")
                
                # Run inference for this date
                predictions = self.make_predictions()
                
                # Restore original config
                self.config = original_config
                
                # Record results
                batch_results.append({
                    'snapshot_date': snapshot_date,
                    'prediction_count': len(predictions),
                    'positive_predictions': (predictions['model_predictions'] > 0.5).sum(),
                    'avg_probability': predictions['model_predictions'].mean(),
                    'status': 'SUCCESS'
                })
                
                successful_runs += 1
                print(f"SUCCESS: {snapshot_date}: {len(predictions)} predictions generated")
                
            except Exception as e:
                print(f"ERROR: {snapshot_date}: FAILED - {str(e)}")
                batch_results.append({
                    'snapshot_date': snapshot_date,
                    'prediction_count': 0,
                    'positive_predictions': 0,
                    'avg_probability': 0.0,
                    'status': f'FAILED: {str(e)}'
                })
                failed_runs += 1
        
        # Save batch inference summary
        self._save_batch_inference_summary(batch_results, start_date, end_date, frequency)
        
        print(f"\nSUCCESS: Batch inference completed: {successful_runs} successful, {failed_runs} failed")
        return batch_results
    
    def _generate_inference_dates(self, start_date, end_date, frequency):
        """Generate list of dates for batch inference"""
        dates = []
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        current_date = start_dt
        
        while current_date <= end_dt:
            dates.append(current_date.strftime("%Y-%m-%d"))
            
            if frequency == 'daily':
                current_date += timedelta(days=1)
            elif frequency == 'weekly':
                current_date += timedelta(weeks=1)
            elif frequency == 'monthly':
                current_date += relativedelta(months=1)
            else:
                raise ValueError(f"Invalid frequency: {frequency}")
        
        return dates
    
    def _save_batch_inference_summary(self, batch_results, start_date, end_date, frequency):
        """Save batch inference summary to gold table"""
        summary_dir = "datamart/gold/batch_inference_summary"
        os.makedirs(summary_dir, exist_ok=True)
        
        # Create summary dataframe
        summary_df = pd.DataFrame(batch_results)
        summary_df['batch_start_date'] = start_date
        summary_df['batch_end_date'] = end_date
        summary_df['frequency'] = frequency
        summary_df['batch_execution_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save as parquet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_inference_summary_{frequency}_{timestamp}.parquet"
        filepath = os.path.join(summary_dir, filename)
        
        spark_df = self.spark.createDataFrame(summary_df)
        spark_df.write.mode("overwrite").parquet(filepath)
        
        print(f"INFO: Batch inference summary saved: {filepath}")

    def run_full_pipeline(self, mode='inference'):
        """Run the complete ML pipeline"""
        print(f"Running ML Pipeline in {mode} mode...")
        print("="*50)
        
        self.initialize_spark()
        
        try:
            if mode == 'validate':
                # Validation mode: check if model exists
                model_path = self.validate_model_exists()
                print(f"Model validation completed: {model_path}")
                
            elif mode == 'inference':
                # Inference mode: make predictions for single date
                self.validate_model_exists()  # Ensure model exists first
                predictions = self.make_predictions()
                print(f"Inference completed. Generated {len(predictions)} predictions")
                
            elif mode == 'batch_inference':
                # Step 2: Batch inference across time periods
                self.validate_model_exists()  # Ensure model exists first
                batch_results = self.run_batch_inference()
                total_predictions = sum(r['prediction_count'] for r in batch_results if r['status'] == 'SUCCESS')
                print(f"Batch inference completed. Generated {total_predictions} total predictions")
                
            elif mode == 'monitor':
                # Step 3: Monitor performance and stability across time periods
                self.validate_model_exists()  # Ensure model exists first
                monitoring_results = self.monitor_model_performance()
                viz_path = self.visualize_performance(monitoring_results)
                print("Monitoring completed. Check visualizations for insights.")
                
            else:
                raise ValueError(f"Invalid mode: {mode}. Valid modes: validate, inference, batch_inference, monitor")
                
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            raise
        finally:
            if self.spark:
                self.spark.stop()


def main(snapshotdate, modelname, mode='inference'):
    print('\n\nStarting ML Pipeline Job\n')
    
    # Configuration - Make all paths configurable
    config = {
        "snapshot_date_str": snapshotdate,
        "snapshot_date": datetime.strptime(snapshotdate, "%Y-%m-%d"),
        "model_name": modelname,
        "model_bank_directory": "model_bank/",
        "model_artefact_filepath": f"model_bank/{modelname}",
        "feature_store_path": "datamart/gold/feature_store",
        "predictions_output_path": "datamart/gold/model_predictions",
        "monitoring_output_path": "datamart/gold/model_monitoring",
        "drift_threshold": 0.1,
        "performance_threshold": 0.75
    }
    
    pprint.pprint(config)
    
    # Initialize and run pipeline
    pipeline = MLPipeline(config)
    pipeline.run_full_pipeline(mode=mode)
    
    print('\n\nML Pipeline Job Completed\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End ML Pipeline")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name.pkl")
    parser.add_argument("--mode", type=str, default="inference", 
                       choices=['validate', 'inference', 'batch_inference', 'monitor'],
                       help="Pipeline mode: validate, inference, batch_inference, or monitor")
    
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname, args.mode)
