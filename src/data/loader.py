"""
Data loading and inspection module for real psychometric dataset (DASS-42).
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np


DATA_URL = "https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip"


def download_and_load_data(raw_dir: str = "data/raw", force_download: bool = False) -> pd.DataFrame:
    """
    Downloads the official Open Source Psychometrics DASS-42 research dataset if not present,
    unzips it, performs quality-control filtering, and returns the cleaned DataFrame.
    """
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "DASS_data_21.02.19.zip")

    if force_download or not os.path.exists(zip_path):
        print(f"[DATA LOADER] Downloading real research dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        print("[DATA LOADER] Download complete.")

    csv_path = None
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f == "data.csv" or (f.endswith(".csv") and "DASS" in f):
                csv_path = os.path.join(root, f)
                break

    if not csv_path or force_download:
        print("[DATA LOADER] Extracting dataset archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if f == "data.csv" or (f.endswith(".csv") and "DASS" in f):
                    csv_path = os.path.join(root, f)
                    break

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not locate data.csv in {raw_dir}")

    # Inspect delimiter (tab vs comma)
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","

    df_raw = pd.read_csv(csv_path, sep=sep)

    # Automated Quality Control Filtering
    # VCL6, VCL9, VCL12 are validity check fake words in the survey.
    valid_mask = (df_raw["VCL6"] == 0) & (df_raw["VCL9"] == 0) & (df_raw["VCL12"] == 0)
    valid_mask &= (df_raw["age"] >= 10) & (df_raw["age"] <= 100)
    df_clean = df_raw[valid_mask].copy()

    return df_clean


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Dynamically inspects the dataset and prints calculated metrics.
    No values are hardcoded.
    """
    num_samples = len(df)
    num_features = df.shape[1]
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate target (Depression Risk score cutoff >= 14)
    dep_items = [f"Q{i}A" for i in [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42] if f"Q{i}A" in df.columns]
    dep_scores = df[dep_items].apply(lambda x: x - 1).sum(axis=1)
    target_series = (dep_scores >= 14).astype(int)
    class_counts = target_series.value_counts().to_dict()

    print("\n" + "="*60)
    print("           REAL DATASET AUTOMATED INSPECTION")
    print("="*60)
    print(f"Dataset loaded successfully")
    print(f"Samples: {num_samples:,}")
    print(f"Features: {num_features}")
    print(f"Missing values: {missing_values}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Numerical features count: {len(num_cols)}")
    print(f"Categorical features count: {len(cat_cols)}")
    print(f"Target variable: 'depression_risk' (DASS-42 Cutoff >= 14)")
    print(f"Classes distribution: Class 1 (Moderate-to-Severe): {class_counts.get(1, 0):,}, Class 0 (Normal/Mild): {class_counts.get(0, 0):,}")
    print("="*60 + "\n")

    return {
        "num_samples": num_samples,
        "num_features": num_features,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numerical_features_count": len(num_cols),
        "categorical_features_count": len(cat_cols),
        "class_counts": class_counts
    }
