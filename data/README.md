# Dataset Documentation: DASS-42 (Depression Anxiety Stress Scales)

## Provenance and Metadata

* **Dataset Name**: Depression Anxiety Stress Scales (DASS-42) Dataset
* **Original Source**: Open Source Psychometrics Project ([openpsychometrics.org](https://openpsychometrics.org/))
* **Direct Download URL**: `https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip`
* **Original Research Paper**: 
  * Lovibond, S.H. & Lovibond, P.F. (1995). *Manual for the Depression Anxiety Stress Scales*. (2nd ed.). Sydney: Psychology Foundation.
  * DOI: [10.1016/0005-7967(94)00075-U](https://doi.org/10.1016/0005-7967(94)00075-U)
* **Access & License**: Public domain for research and educational purposes with proper attribution.
* **Participant Count**: 39,775 raw responses (34,576 after quality control filtering)
* **Variables**: 172 columns (42 survey items, item timings, item positions, TIPI Big Five personality inventory, vocabulary checks, and demographic variables)

---

## Dataset Description

The DASS-42 is a 42-item self-report instrument designed to measure the three related negative emotional states of **Depression**, **Anxiety**, and **Stress**.

### Psychometric Subscales

1. **Depression Scale (D)**: Evaluates dysphoria, hopelessness, devaluation of life, self-deprecation, lack of interest/involvement, anhedonia, and inertia.
   * *Items*: Q3, Q5, Q10, Q13, Q16, Q17, Q21, Q24, Q26, Q31, Q34, Q37, Q38, Q42.
2. **Anxiety Scale (A)**: Evaluates autonomic arousal, skeletal muscle effects, situational anxiety, and subjective experience of anxious affect.
   * *Items*: Q2, Q4, Q7, Q9, Q15, Q19, Q20, Q23, Q25, Q28, Q30, Q36, Q40, Q41.
3. **Stress Scale (S)**: Evaluates difficulty relaxing, nervous arousal, being easily upset/agitated, irritable/over-reactive, and impatient.
   * *Items*: Q1, Q6, Q8, Q11, Q12, Q14, Q18, Q22, Q27, Q29, Q32, Q33, Q35, Q39.

---

## Quality Control and Preprocessing

Raw survey data can contain careless or automated responses. The following automated cleaning steps are applied in `src/data/loader.py`:

1. **Validity Word Checks (`VCL6`, `VCL9`, `VCL12`)**:
   * The survey includes fake control words ("cuivocal", "florted", "verdid"). Participants who checked any of these fake words were omitted as careless responders.
2. **Age Sanity Filtering**:
   * Participants with recorded ages $< 10$ or $> 100$ were removed.
3. **Likert Rescaling**:
   * Raw survey items are recorded on a 1–4 scale. Standard DASS scoring subtracts 1 so items range from 0 (*Did not apply*) to 3 (*Applied very much*).

---

## Target Variable Selection (`depression_risk`)

* **Derivation**: Standard Lovibond & Lovibond (1995) cut-offs for DASS-42 Depression Subscale (sum of 14 items, score range 0 to 42):
  * **Normal**: 0 – 9
  * **Mild**: 10 – 13
  * **Moderate**: 14 – 20
  * **Severe**: 21 – 27
  * **Extremely Severe**: 28+
* **Binary Target (`depression_risk`)**:
  * `1` = **Moderate-to-Severe Depression Risk** (Score $\ge 14$)
  * `0` = **Normal / Mild Depression** (Score $< 14$)

---

## Ethics and Privacy Notice

* The dataset is fully anonymized at the source and contains zero personally identifiable information (PII).
* Raw data files are ignored by version control (`.gitignore`) to avoid unauthorized data redistribution.
* This dataset is used strictly for academic research in Explainable AI (XAI) and does **not** constitute a clinical diagnostic tool.
