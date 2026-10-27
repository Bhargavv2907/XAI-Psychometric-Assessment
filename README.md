<<<<<<< HEAD
# Explainable AI Framework for Psychometric Assessment

An open research implementation of an **Explainable AI (XAI) framework for psychometric and mental health assessment** built on real empirical research dataset responses (DASS-42).

> **Evaluation Phase**: First Screening / Evaluation 1  
> **Status**: Working Research Foundation  

---

## 📌 Project Overview

Traditional AI models used in psychometric assessment and mental health screening often function as black boxes, providing risk scores without explaining the underlying clinical factors or item contributions. 

This project develops an **Explainable AI (XAI) Framework** that integrates model-agnostic explanation techniques (**SHAP** and **LIME**) with machine learning classifiers trained on validated psychometric assessment data. The goal is to produce transparent, feature-level, and human-interpretable explanations suitable for psychological researchers and clinicians.

---

## 🎯 Three Core Project Tasks

1. **Task 1**: Develop an XAI framework for psychometric assessment by integrating explainability techniques with AI-based psychometric assessment models.
2. **Task 2**: Implement and compare explainability techniques:
   * [x] **SHAP** *(Implemented in Evaluation 1)*
   * [x] **LIME** *(Implemented in Evaluation 1)*
   * [ ] **Integrated Gradients** *(Planned extension point)*
   * [ ] **Grad-CAM** *(Planned extension point)*
   * [ ] **Attention-based explanations** *(Planned extension point)*
   * [ ] **Counterfactual Explanations** *(Planned extension point)*
3. **Task 3**: Develop multimodal explainability mechanisms for questionnaire responses, behavioural data, text transcriptions, speech acoustics, and physiological signals.

---

## 📊 Dataset Provenance & Selection

* **Dataset Name**: Depression Anxiety Stress Scales (DASS-42) Dataset
* **Original Source**: Open Source Psychometrics Project ([openpsychometrics.org](https://openpsychometrics.org/))
* **Original Research Paper**: Lovibond, S.H. & Lovibond, P.F. (1995). *Manual for the Depression Anxiety Stress Scales* (2nd ed.). Sydney: Psychology Foundation. DOI: [10.1016/0005-7967(94)00075-U](https://doi.org/10.1016/0005-7967(94)00075-U)
* **License / Access**: Open domain for public research and educational use.
* **Participant Count**: 39,775 total raw respondents ($34,576$ after quality-control filtering).
* **Target Variable**: Binary classification `depression_risk` based on standard Lovibond & Lovibond clinical cut-offs (DASS Depression Score $\ge 14$: Moderate-to-Severe Depression Risk).

---

## 🏗️ Evaluation 1 Implementation vs Future Roadmap

### ✅ IMPLEMENTED IN EVALUATION 1

```text
✓ Real research dataset integration (DASS-42)
✓ Automated quality control & dataset inspection
✓ Preprocessing & Likert rescaling to standard 0-3 item scores
✓ Human-readable item mapping (e.g. "Q3: No positive feeling at all")
✓ Stratified Train/Test splitting
✓ Baseline Machine Learning Models (Random Forest, XGBoost, Logistic Regression)
✓ Dynamic evaluation metrics calculation (Accuracy, Precision, Recall, F1, ROC-AUC)
✓ Real SHAP global feature importance summary & bar plots
✓ Real SHAP local participant sample waterfall/bar explanation
✓ Real LIME local tabular feature contribution explanation & interactive HTML report
✓ Modular XAI Framework Manager architecture
✓ Visualizations saved automatically to outputs/ directory
```

### 🔮 PLANNED FOR FUTURE PHASES

```text
○ Integrated Gradients for neural network baseline models
○ Grad-CAM for visual/imaging psychometric representations
○ Attention weight explanations for Transformer NLP text/speech models
○ Counterfactual explanations for actionable clinical recourse
○ Multimodal feature fusion (Questionnaire + Behavioural + Text + Speech + Physiological)
○ Modality-level attribution explanations
○ Quantitative comparative evaluation across XAI techniques
```

---

## 📁 Repository Structure

```text
xai-psychometric-assessment/
│
├── data/
│   ├── raw/                   # Raw DASS-42 dataset zip & extracted files (Git ignored)
│   ├── processed/             # Cleaned preprocessed datasets
│   └── README.md              # Detailed dataset metadata & provenance documentation
│
├── notebooks/                 # Exploratory research notebooks
│
├── src/
│   ├── data/
│   │   ├── loader.py          # Downloads, extracts & automatically inspects dataset
│   │   └── preprocessing.py   # Cleans data, maps question text & builds target
│   │
│   ├── models/
│   │   └── baseline_model.py  # Trains baseline classifiers (Random Forest, XGBoost)
│   │
│   ├── explainers/
│   │   ├── shap_explainer.py  # Computes global & local SHAP explanations
│   │   ├── lime_explainer.py  # Computes LIME tabular explanations & HTML report
│   │   ├── integrated_gradients.py  # Extension stub (NotImplementedError)
│   │   ├── grad_cam.py             # Extension stub (NotImplementedError)
│   │   ├── attention_explainer.py  # Extension stub (NotImplementedError)
│   │   └── counterfactual.py       # Extension stub (NotImplementedError)
│   │
│   ├── multimodal/
│   │   └── architecture.py    # Conceptual blueprint for future multimodal fusion
│   │
│   └── evaluation/
│       └── metrics.py         # Dynamic calculation of metrics & ROC/Confusion plots
│
├── outputs/
│   ├── figures/               # Confusion matrix & ROC curve plots
│   ├── shap/                  # SHAP global summary, bar & local sample plots
│   └── lime/                  # LIME local sample plots & HTML reports
│
├── main.py                    # Main pipeline entrypoint script
├── requirements.txt           # Dependency requirements
└── README.md                  # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Main Evaluation 1 Pipeline
```bash
python main.py
```

### Optional Arguments
To specify a different baseline model architecture or test participant index:
```bash
python main.py --model xgboost --sample_idx 5
```

---

## ⚖️ Academic Integrity and Ethics Statement

1. **Empirical Grounding**: Zero synthetic, fake, or manually hardcoded data/metrics are used. All results are derived dynamically from the official Open Source Psychometrics Project DASS-42 dataset.
2. **Participant Privacy**: The dataset is completely anonymized at the source. Raw data files are kept local and excluded from repository commits (`.gitignore`).
3. **Non-Clinical Scope**: This software is an **academic research prototype** designed to study interpretability in machine learning. It is **not** a clinical diagnostic tool or medical software.
=======
# XAI
>>>>>>> 87cf5d924a868392e428e265227272804a989a41
