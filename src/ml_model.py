import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

def run_ml_pipeline():
    print("=" * 75)
    print("STARTING MODULE 4: SUPERVISED ML MODELING PIPELINE")
    print("=" * 75)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "outputs", "data")
    reports_dir = os.path.join(base_dir, "outputs", "reports")
    figures_dir = os.path.join(base_dir, "outputs", "figures")
    models_dir = os.path.join(base_dir, "models")
    
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Partitioned Datasets
    train_path = os.path.join(data_dir, "ml_train_features.csv")
    val_path = os.path.join(data_dir, "ml_val_features.csv")
    test_path = os.path.join(data_dir, "ml_test_features.csv")
    
    print(f"Loading datasets from {data_dir}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train set: {len(train_df):,} rows, {train_df.shape[1]} columns")
    print(f"Val set:   {len(val_df):,} rows, {val_df.shape[1]} columns")
    print(f"Test set:  {len(test_df):,} rows, {test_df.shape[1]} columns")
    
    # 2. Approved Feature Selection
    # Load 33-feature manifest and strictly exclude target and audit-identified leakage
    manifest_path = os.path.join(reports_dir, "ml_feature_manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    raw_manifest_features = manifest['feature_list']
    
    # Exclude route_target_enc_time due to in-sample leakage on train & constant on val/test
    excluded_audit_features = ['route_target_enc_time', 'travel_time_sec', 'next_arr_sec']
    selected_features = [f for f in raw_manifest_features if f not in excluded_audit_features]
    
    target_col = 'travel_time_sec'
    
    print(f"\nTarget Variable: '{target_col}'")
    print(f"Features selected for training ({len(selected_features)}):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
        
    print(f"\nExplicitly excluded: {excluded_audit_features}")
    
    X_train = train_df[selected_features]
    y_train = train_df[target_col]
    
    X_val = val_df[selected_features]
    y_val = val_df[target_col]
    
    X_test = test_df[selected_features]
    y_test = test_df[target_col]
    
    # 3. Model Candidates Definition
    models = {
        'DummyRegressor (Mean Baseline)': DummyRegressor(strategy='mean'),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=12,
            learning_rate=0.08,
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }
    
    # 4. Training and Validation Evaluation
    print("\n" + "=" * 75)
    print("TRAINING CANDIDATE MODELS & EVALUATING ON VALIDATION SET")
    print("=" * 75)
    
    val_results = []
    trained_val_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        t_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t_start
        
        t_pred_start = time.time()
        preds_val = model.predict(X_val)
        pred_time = time.time() - t_pred_start
        
        mae_val = mean_absolute_error(y_val, preds_val)
        rmse_val = root_mean_squared_error(y_val, preds_val)
        r2_val = r2_score(y_val, preds_val)
        medae_val = median_absolute_error(y_val, preds_val)
        
        trained_val_models[name] = model
        
        res = {
            'model_name': name,
            'train_time_sec': round(fit_time, 3),
            'val_pred_time_sec': round(pred_time, 3),
            'val_mae_sec': round(mae_val, 3),
            'val_rmse_sec': round(rmse_val, 3),
            'val_r2': round(r2_val, 5),
            'val_medae_sec': round(medae_val, 3)
        }
        val_results.append(res)
        
        print(f"  Fit Time: {fit_time:.2f}s | Val Pred Time: {pred_time:.3f}s")
        print(f"  Val MAE:  {mae_val:.2f}s | Val RMSE: {rmse_val:.2f}s | Val R2: {r2_val:.5f} | Val MedAE: {medae_val:.2f}s")
        
    val_df_results = pd.DataFrame(val_results)
    
    # Sort models by Validation MAE (ascending)
    val_df_results = val_df_results.sort_values(by='val_mae_sec', ascending=True).reset_index(drop=True)
    
    print("\n" + "-" * 75)
    print("VALIDATION BENCHMARK SUMMARY (Ranked by Validation MAE):")
    print("-" * 75)
    print(val_df_results.to_string(index=False))
    
    # 5. Model Selection Based on Validation Performance
    best_model_name = val_df_results.iloc[0]['model_name']
    best_val_mae = val_df_results.iloc[0]['val_mae_sec']
    best_val_r2 = val_df_results.iloc[0]['val_r2']
    print(f"\nSELECTED BEST MODEL: '{best_model_name}' (Val MAE: {best_val_mae:.2f}s, Val R2: {best_val_r2:.5f})")
    
    # 6. Evaluate Selected Model ONCE on the completely unseen TEST Set
    print("\n" + "=" * 75)
    print("EVALUATING SELECTED MODEL ON UNSEEN TEST SET")
    print("=" * 75)
    
    selected_model = trained_val_models[best_model_name]
    baseline_model = trained_val_models['DummyRegressor (Mean Baseline)']
    
    t_test_start = time.time()
    test_preds = selected_model.predict(X_test)
    test_pred_time = time.time() - t_test_start
    
    test_baseline_preds = baseline_model.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_medae = median_absolute_error(y_test, test_preds)
    
    # Mean Absolute Percentage Error (avoid division by 0)
    test_mape = np.mean(np.abs((y_test - test_preds) / np.maximum(y_test, 1.0))) * 100.0
    
    baseline_test_mae = mean_absolute_error(y_test, test_baseline_preds)
    baseline_test_rmse = root_mean_squared_error(y_test, test_baseline_preds)
    baseline_test_r2 = r2_score(y_test, test_baseline_preds)
    
    print(f"Final Model: '{best_model_name}'")
    print(f"  Test MAE:   {test_mae:.2f} s")
    print(f"  Test RMSE:  {test_rmse:.2f} s")
    print(f"  Test R2:    {test_r2:.5f}")
    print(f"  Test MedAE: {test_medae:.2f} s")
    print(f"  Test MAPE:  {test_mape:.2f} %")
    print(f"  Inference Latency: {test_pred_time*1000 / len(X_test):.4f} ms per segment")
    
    print("\nComparison Against Dummy Mean Baseline:")
    print(f"  Baseline Test MAE:  {baseline_test_mae:.2f} s -> Selected Model: {test_mae:.2f} s (Error reduced by {((baseline_test_mae - test_mae) / baseline_test_mae)*100:.1f}%)")
    print(f"  Baseline Test RMSE: {baseline_test_rmse:.2f} s -> Selected Model: {test_rmse:.2f} s (Error reduced by {((baseline_test_rmse - test_rmse) / baseline_test_rmse)*100:.1f}%)")
    print(f"  Baseline Test R2:   {baseline_test_r2:.5f} -> Selected Model: {test_r2:.5f}")
    
    # 7. Generate Evaluation Plots
    print("\n" + "=" * 75)
    print("GENERATING EVALUATION PLOTS")
    print("=" * 75)
    
    residuals = y_test - test_preds
    
    # Plot 1: Actual vs Predicted Travel Time
    plt.figure(figsize=(9, 6), dpi=300)
    # Subsample for clean scatter visualization if > 5000 points
    sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(5000, len(y_test)), replace=False)
    y_sub_test = y_test.iloc[sample_idx]
    sub_preds = test_preds[sample_idx]
    
    plt.scatter(y_sub_test, sub_preds, alpha=0.35, color='#1f77b4', edgecolors='none', s=25, label='Test Segments')
    
    max_val = max(y_sub_test.quantile(0.999), sub_preds.max())
    plt.plot([0, max_val], [0, max_val], color='#d62728', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
    
    plt.title(f"Actual vs Predicted Segment Travel Time ({best_model_name})\nTest Set (Sampled 5,000 Segments)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Actual Travel Time (seconds)", fontsize=11)
    plt.ylabel("Predicted Travel Time (seconds)", fontsize=11)
    plt.xlim(0, min(1500, max_val))
    plt.ylim(0, min(1500, max_val))
    plt.legend(frameon=True, facecolor='white', loc='upper left')
    
    # Annotation box
    metric_box = f"Test MAE: {test_mae:.2f}s\nTest RMSE: {test_rmse:.2f}s\nTest R²: {test_r2:.5f}\nTest MedAE: {test_medae:.2f}s"
    plt.gca().text(0.95, 0.05, metric_box, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#ccc'))
    
    actual_vs_pred_fig = os.path.join(figures_dir, "actual_vs_predicted_travel_time.png")
    plt.savefig(actual_vs_pred_fig, bbox_inches='tight')
    plt.close()
    print(f"Saved: {actual_vs_pred_fig}")
    
    # Plot 2: Residual / Error Distribution
    plt.figure(figsize=(9, 5), dpi=300)
    # Clip residuals for visualization to focus on bulk distribution (-50 to +50 seconds)
    clipped_res = np.clip(residuals, -30, 30)
    
    sns.histplot(clipped_res, bins=60, kde=True, color='#2ca02c', edgecolor='white', alpha=0.7)
    plt.axvline(0, color='#d62728', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.axvline(residuals.mean(), color='#1f77b4', linestyle=':', linewidth=1.5, label=f'Mean Error ({residuals.mean():.2f}s)')
    
    plt.title(f"Residual Error Distribution (Actual - Predicted)\n{best_model_name} on Test Set", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Prediction Error (seconds, clipped to [-30s, +30s])", fontsize=11)
    plt.ylabel("Segment Count", fontsize=11)
    plt.legend(frameon=True, facecolor='white')
    
    residual_fig = os.path.join(figures_dir, "residual_error_distribution.png")
    plt.savefig(residual_fig, bbox_inches='tight')
    plt.close()
    print(f"Saved: {residual_fig}")
    
    # Plot 3: Feature Importance
    if hasattr(selected_model, 'feature_importances_'):
        importances = selected_model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'feature': selected_features,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(10, 8), dpi=300)
        top_n = min(15, len(feat_imp_df))
        top_feats = feat_imp_df.head(top_n)
        
        palette = sns.color_palette("Blues_r", n_colors=top_n)
        bars = plt.barh(range(top_n), top_feats['importance'].values[::-1], color=palette[::-1], edgecolor='none', height=0.7)
        plt.yticks(range(top_n), top_feats['feature'].values[::-1], fontsize=10)
        
        plt.title(f"Top {top_n} Most Important Features ({best_model_name})", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Feature Importance (Gini / Split Gain)", fontsize=11)
        
        # Add labels on bars
        for bar in bars:
            val = bar.get_width()
            plt.gca().text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                           ha='left', va='center', fontsize=9, color='#333')
            
        plt.xlim(0, max(top_feats['importance']) * 1.15)
        
        feat_imp_fig = os.path.join(figures_dir, "feature_importance.png")
        plt.savefig(feat_imp_fig, bbox_inches='tight')
        plt.close()
        print(f"Saved: {feat_imp_fig}")
    else:
        feat_imp_df = pd.DataFrame({'feature': selected_features, 'importance': [None]*len(selected_features)})
    
    # 8. Save Trained Model Artifacts
    model_save_path = os.path.join(models_dir, "best_travel_time_model.joblib")
    joblib.dump(selected_model, model_save_path)
    print(f"\nSaved trained model to: {model_save_path}")
    
    # Save feature metadata
    features_meta_path = os.path.join(models_dir, "model_features.json")
    with open(features_meta_path, 'w') as f:
        json.dump({
            "selected_model": best_model_name,
            "feature_count": len(selected_features),
            "features": selected_features,
            "target": target_col,
            "training_rows": len(X_train),
            "test_rows": len(X_test)
        }, f, indent=2)
    print(f"Saved model feature metadata to: {features_meta_path}")
    
    # 9. Save Evaluation Results (CSV and JSON)
    comp_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    val_df_results.to_csv(comp_csv_path, index=False)
    print(f"Saved model comparison to: {comp_csv_path}")
    
    test_eval_metrics = {
        "selected_model": best_model_name,
        "validation_metrics": {
            "val_mae_sec": float(best_val_mae),
            "val_rmse_sec": float(val_df_results.iloc[0]['val_rmse_sec']),
            "val_r2": float(best_val_r2),
            "val_medae_sec": float(val_df_results.iloc[0]['val_medae_sec'])
        },
        "test_metrics": {
            "test_mae_sec": float(round(test_mae, 3)),
            "test_rmse_sec": float(round(test_rmse, 3)),
            "test_r2": float(round(test_r2, 5)),
            "test_medae_sec": float(round(test_medae, 3)),
            "test_mape_percent": float(round(test_mape, 2)),
            "test_inference_latency_ms_per_sample": float(round((test_pred_time*1000 / len(X_test)), 4))
        },
        "baseline_comparison": {
            "baseline_model": "DummyRegressor (Training Mean)",
            "baseline_test_mae_sec": float(round(baseline_test_mae, 3)),
            "baseline_test_rmse_sec": float(round(baseline_test_rmse, 3)),
            "baseline_test_r2": float(round(baseline_test_r2, 5)),
            "mae_reduction_percent": float(round(((baseline_test_mae - test_mae) / baseline_test_mae) * 100.0, 2)),
            "rmse_reduction_percent": float(round(((baseline_test_rmse - test_rmse) / baseline_test_rmse) * 100.0, 2))
        },
        "top_features": feat_imp_df.head(10).to_dict(orient='records')
    }
    
    test_metrics_json_path = os.path.join(reports_dir, "model_evaluation_metrics.json")
    with open(test_metrics_json_path, 'w') as f:
        json.dump(test_eval_metrics, f, indent=2)
    print(f"Saved test evaluation metrics to: {test_metrics_json_path}")
    
    print("\n" + "=" * 75)
    print("MODULE 4 ML MODELING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == '__main__':
    run_ml_pipeline()
