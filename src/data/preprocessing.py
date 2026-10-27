"""
Data preprocessing module for DASS-42 psychometric assessment.
Maps raw questionnaire items to human-readable text descriptions, computes target labels,
and creates clean train/test data splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List


# Map raw survey columns to human-readable psychological question descriptions
ITEM_LABEL_MAP: Dict[str, str] = {
    "Q1A": "Q1: Upset by trivial things (Stress)",
    "Q2A": "Q2: Dryness of mouth (Anxiety)",
    "Q3A": "Q3: No positive feeling at all (Depression)",
    "Q4A": "Q4: Breathing difficulty (Anxiety)",
    "Q5A": "Q5: Hard to work up initiative (Depression)",
    "Q6A": "Q6: Over-reacted to situations (Stress)",
    "Q7A": "Q7: Feeling of shakiness (Anxiety)",
    "Q8A": "Q8: Found it difficult to relax (Stress)",
    "Q9A": "Q9: Situations making me anxious (Anxiety)",
    "Q10A": "Q10: Nothing to look forward to (Depression)",
    "Q11A": "Q11: Getting upset easily (Stress)",
    "Q12A": "Q12: Using a lot of nervous energy (Stress)",
    "Q13A": "Q13: Felt sad and depressed (Depression)",
    "Q14A": "Q14: Impatient when delayed (Stress)",
    "Q15A": "Q15: Feeling of faintness (Anxiety)",
    "Q16A": "Q16: Lost interest in everything (Depression)",
    "Q17A": "Q17: Felt wasn't worth much as person (Depression)",
    "Q18A": "Q18: Felt touchy / sensitive (Stress)",
    "Q19A": "Q19: Perspired noticeably (Anxiety)",
    "Q20A": "Q20: Felt scared without good reason (Anxiety)",
    "Q21A": "Q21: Felt life wasn't worthwhile (Depression)",
    "Q22A": "Q22: Hard to wind down (Stress)",
    "Q23A": "Q23: Difficulty in swallowing (Anxiety)",
    "Q24A": "Q24: Couldn't get enjoyment from things (Depression)",
    "Q25A": "Q25: Aware of heart action/palpitations (Anxiety)",
    "Q26A": "Q26: Felt down-hearted and blue (Depression)",
    "Q27A": "Q27: Found very irritable (Stress)",
    "Q28A": "Q28: Felt close to panic (Anxiety)",
    "Q29A": "Q29: Hard to calm down (Stress)",
    "Q30A": "Q30: Feared thrown by unfamiliar task (Anxiety)",
    "Q31A": "Q31: Unable to get enthusiastic (Depression)",
    "Q32A": "Q32: Difficult to tolerate interruptions (Stress)",
    "Q33A": "Q33: State of nervous tension (Stress)",
    "Q34A": "Q34: Felt pretty worthless (Depression)",
    "Q35A": "Q35: Intolerant of delays (Stress)",
    "Q36A": "Q36: Felt terrified (Anxiety)",
    "Q37A": "Q37: Nothing in future to be hopeful (Depression)",
    "Q38A": "Q38: Felt life was meaningless (Depression)",
    "Q39A": "Q39: Found getting agitated (Stress)",
    "Q40A": "Q40: Worried about panic in public (Anxiety)",
    "Q41A": "Q41: Experienced trembling (Anxiety)",
    "Q42A": "Q42: Hard to work up initiative (Depression)",
    "TIPI1": "Big5: Extraverted, enthusiastic",
    "TIPI2": "Big5: Critical, quarrelsome",
    "TIPI3": "Big5: Dependable, self-disciplined",
    "TIPI4": "Big5: Anxious, easily upset",
    "TIPI5": "Big5: Open to new experiences",
    "TIPI6": "Big5: Reserved, quiet",
    "TIPI7": "Big5: Sympathetic, warm",
    "TIPI8": "Big5: Disorganized, careless",
    "TIPI9": "Big5: Calm, emotionally stable",
    "TIPI10": "Big5: Conventional, uncreative",
    "age": "Demographic: Age",
    "education": "Demographic: Education Level",
    "gender": "Demographic: Gender",
    "urban": "Demographic: Childhood Area"
}


def preprocess_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Preprocesses real DASS-42 dataframe:
    1. Converts 1-4 Likert scales to 0-3 standard scores.
    2. Calculates standard DASS Depression score & defines binary target 'depression_risk'.
    3. Selects predictor features (Questionnaire items + Personality + Demographics).
    4. Renames columns to human-readable psychological descriptions for XAI interpretability.
    5. Splits into stratified train and test sets.
    """
    # Depression subscale item columns (raw)
    dep_raw_items = [f"Q{i}A" for i in [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]]
    
    # 0-3 scaled depression sum score
    dep_score = df[dep_raw_items].apply(lambda col: col - 1).sum(axis=1)
    
    # Target: 1 if Depression Score >= 14 (Moderate to Severe), 0 if < 14 (Normal / Mild)
    y = (dep_score >= 14).astype(int)
    y.name = "depression_risk"

    # Select feature columns: item responses (Q1A..Q42A), personality (TIPI1..10), demographics (age, education, gender, urban)
    feature_cols = [c for c in df.columns if (c.startswith("Q") and c.endswith("A")) or c.startswith("TIPI") or c in ["age", "education", "gender", "urban"]]
    
    # Create feature dataframe
    X = df[feature_cols].copy()
    
    # Standardize item scores to 0-3 for Q items
    q_cols = [c for c in X.columns if c.startswith("Q") and c.endswith("A")]
    X[q_cols] = X[q_cols].apply(lambda col: col - 1)
    
    # Handle missing values if any
    X = X.fillna(X.median())

    # Map column names to human-readable feature labels
    readable_cols = [ITEM_LABEL_MAP.get(col, col) for col in X.columns]
    X.columns = readable_cols

    # Train/Test Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, list(X.columns)
