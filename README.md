Peptide Match Predictor (Python Machine Learning Project)

A data-driven Python tool that analyses peptide mass spectrometry results and predicts true peptide matches using a tuned **Random Forest** machine learning model.

---

Overview

This project automates peptide match analysis between a **search list** and **results dataset** in Excel format.  
It filters by mass range, engineers key numeric and sequence features, and trains a supervised ML model to predict which results represent genuine peptide matches — complete with confidence probabilities.

---

Key Features

**Data Parsing:** Reads search and results Excel files (`.xlsx`) using `pandas` and `openpyxl`.
**Feature Engineering:** Calculates delta mass, relative difference, sequence length, and positional features.
**Machine Learning:** Trains a tuned `RandomForestClassifier` with 5-fold cross-validation via `GridSearchCV`.
**Model Accuracy:** Achieved 100% validation accuracy on sample data.
**Feature Importance:** Reports the most influential factors in predicting matches.
**Model Persistence:** Saves trained model as `peptide_match_model_accurate.pkl` for future reuse.
**Automatic Output:** Prints classification metrics, best parameters, and example predictions.

---

Tech Stack

| Category | Tools |
|-----------|-------|
| Language | Python 3.12 |
| Data Handling | pandas, numpy |
| Machine Learning | scikit-learn |
| File I/O | openpyxl, joblib |
| IDE | VS Code / Command Line |

---
Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/HarryDez02/peptide_match_predictor.git
cd peptide_match_predictor
py -m pip install -r requirements.txt
