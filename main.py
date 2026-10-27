"""
Explainable AI Framework for Psychometric Assessment
Complete Orchestration Pipeline for Tasks 1, 2, and 3.

Execution Pipeline:
  1. Load real research dataset (DASS-42) & automated quality control inspection
  2. Preprocess features, map Likert items, and split stratified train/test sets
  3. Train baseline or neural AI model (Random Forest, XGBoost, MLP, Conv1D, Transformer)
  4. Evaluate model performance (Accuracy, Precision, Recall, F1, ROC-AUC)
  5. Execute all 6 XAI Explainers:
       - SHAP (SHapley Additive exPlanations)
       - LIME (Local Interpretable Model-agnostic Explanations)
       - Integrated Gradients (Axiomatic path integrals)
       - Grad-CAM (1D Convolutional Class Activation Mapping)
       - Attention-Based Explanations (Transformer Self-Attention)
       - Counterfactual Recourse (Actionable clinical intervention)
  6. Task 3: Multimodal Psychometric Fusion & Explainability (5 data streams)
  7. Task 2: Quantitative XAI Comparative Evaluation (Faithfulness, Sparsity, Concordance)
  8. Interactive Web Dashboard generation
"""

import sys
import os
import argparse

# Ensure local src module is in path
sys.path.insert(0, os.path.abspath("."))

from src.data.loader import download_and_load_data, inspect_dataset
from src.data.preprocessing import preprocess_data
from src.models.baseline_model import train_baseline_model
from src.evaluation.metrics import evaluate_model
from src.explainers import (
    generate_shap_explanations,
    generate_lime_explanations,
    generate_integrated_gradients_explanations,
    generate_gradcam_explanations,
    generate_attention_explanations,
    generate_counterfactual_explanations,
    XAIFrameworkManager
)
from src.multimodal.architecture import MultimodalXAIArchitecture
from src.evaluation.xai_comparison import run_xai_comparison_benchmark
from src.dashboard.dashboard_generator import generate_interactive_dashboard


def main():
    parser = argparse.ArgumentParser(description="Run Complete XAI Psychometric Assessment Pipeline (Tasks 1, 2, & 3)")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "xgboost", "logistic_regression", "neural_mlp", "conv1d", "transformer"],
                        help="Machine learning model architecture")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Test participant index to generate local XAI explanations for")
    parser.add_argument("--skip_multimodal", action="store_true",
                        help="Skip Task 3 multimodal pipeline if set")
    parser.add_argument("--skip_benchmark", action="store_true",
                        help="Skip Task 2 comparative XAI benchmark if set")
    parser.add_argument("--skip_dashboard", action="store_true",
                        help="Skip HTML dashboard generation if set")
    args = parser.parse_args()

    print("\n" + "="*75)
    print("   EXPLAINABLE AI FRAMEWORK FOR PSYCHOMETRIC ASSESSMENT (FULL PIPELINE)")
    print("                 TASKS 1, 2, & 3 COMPREHENSIVE EXECUTION")
    print("="*75 + "\n")

    # Step 1: Download & Load Real Research Dataset
    df_clean = download_and_load_data(raw_dir="data/raw", force_download=False)

    # Step 2: Automated Dataset Inspection
    dataset_stats = inspect_dataset(df_clean)

    # Step 3: Data Preprocessing & Stratified Train/Test Split
    print("[PIPELINE] Preprocessing survey responses, mapping item descriptions & building target...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        df_clean, test_size=0.2, random_state=42
    )
    print(f"[PIPELINE] Training samples: {len(X_train):,} | Testing samples: {len(X_test):,}")
    print(f"[PIPELINE] Total features: {len(feature_names)}")

    # Step 4: Model Training
    model = train_baseline_model(X_train, y_train, model_type=args.model, random_state=42)

    # Step 5: Model Performance Evaluation
    metrics = evaluate_model(model, X_test, y_test, output_dir="outputs/figures")

    # Step 6: Execute All 6 XAI Explainers
    print("\n" + "="*70)
    print("              EXECUTING ALL 6 EXPLAINABILITY TECHNIQUES")
    print("="*70)

    # 1. SHAP
    shap_results = generate_shap_explanations(
        model, X_train, X_test, feature_names, output_dir="outputs/shap", sample_idx=args.sample_idx
    )

    # 2. LIME
    lime_results = generate_lime_explanations(
        model, X_train, X_test, feature_names, output_dir="outputs/lime", sample_idx=args.sample_idx
    )

    # 3. Integrated Gradients
    ig_results = generate_integrated_gradients_explanations(
        model, X_train, X_test, feature_names, output_dir="outputs/ig", sample_idx=args.sample_idx
    )

    # 4. Grad-CAM
    gradcam_results = generate_gradcam_explanations(
        model, X_train, X_test, feature_names, output_dir="outputs/gradcam", sample_idx=args.sample_idx
    )

    # 5. Attention-Based Explanations
    attention_results = generate_attention_explanations(
        model, X_train, X_test, feature_names, output_dir="outputs/attention", sample_idx=args.sample_idx
    )

    # 6. Counterfactual Recourse
    cf_results = generate_counterfactual_explanations(
        model, X_train, X_test, feature_names, output_dir="outputs/counterfactual", sample_idx=args.sample_idx
    )

    # Display Framework Manager Status
    xai_manager = XAIFrameworkManager()
    xai_manager.list_status()

    # Step 7: Task 3 Multimodal Psychometric Fusion & Explainability
    if not args.skip_multimodal:
        multimodal_arch = MultimodalXAIArchitecture()
        mm_results = multimodal_arch.run_pipeline(
            df_clean, sample_idx=args.sample_idx, output_dir="outputs/multimodal"
        )

    # Step 8: Task 2 Quantitative XAI Comparative Evaluation
    if not args.skip_benchmark:
        benchmark_results = run_xai_comparison_benchmark(
            model, X_train, X_test, feature_names, output_dir="outputs/comparison", sample_idx=args.sample_idx
        )

    # Step 9: Interactive Web Dashboard Generation
    if not args.skip_dashboard:
        dashboard_path = generate_interactive_dashboard(
            metrics=metrics, sample_idx=args.sample_idx, output_path="outputs/dashboard/index.html"
        )

    print("\n" + "="*75)
    print("      [SUCCESS] COMPLETE XAI FRAMEWORK PIPELINE EXECUTED SUCCESSFULLY!")
    print("="*75)
    print("Summary of Generated Deliverables:")
    print("  * Evaluation Metrics:     outputs/figures/ (Confusion Matrix, ROC Curve)")
    print("  * SHAP Visualizations:    outputs/shap/ (Summary, Bar Plot, Local Sample)")
    print("  * LIME Reports:           outputs/lime/ (Local Plot, Interactive HTML)")
    print("  * Integrated Gradients:   outputs/ig/ (Local Sample, Global Cohort)")
    print("  * Grad-CAM Maps:          outputs/gradcam/ (1D Item Activation & Subscales)")
    print("  * Attention Matrices:     outputs/attention/ (Subscale Matrix, Symptom Pairs)")
    print("  * Counterfactual Recourse:outputs/counterfactual/ (Actionable Intervention)")
    print("  * Multimodal XAI Engine:  outputs/multimodal/ (Radar Chart, Contribution Bars)")
    print("  * Quantitative Benchmark: outputs/comparison/ (Concordance Matrix, Deletion Curves)")
    print("  * Interactive Dashboard:  outputs/dashboard/index.html")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
