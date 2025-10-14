import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def prepare_dataset(search_path, results_path):
    """Create a merged dataset with engineered features for ML."""
    search_df = pd.read_excel(search_path)
    results_df = pd.read_excel(results_path)

    # Convert to numeric safely
    for col in ["Mass", "Acceptable range", "Unnamed: 5"]:
        if col in search_df.columns:
            search_df[col] = pd.to_numeric(search_df[col], errors="coerce")
    if "Mass" in results_df.columns:
        results_df["Mass"] = pd.to_numeric(results_df["Mass"], errors="coerce")

    matches = []
    for _, row in search_df.iterrows():
        lower = row.get("Acceptable range")
        upper = row.get("Unnamed: 5")
        tgt_mass = row.get("Mass")
        if pd.isna(lower) or pd.isna(upper) or pd.isna(tgt_mass):
            continue
        seq = row.get("Sequence")
        site = row.get("Site")
        seq_loc = row.get("Seq Loc")

        mask = results_df["Mass"].between(lower, upper, inclusive="both")
        subset = results_df.loc[mask].copy()
        if subset.empty:
            continue

        subset.insert(0, "Matched Sequence", seq)
        subset.insert(1, "Matched Seq Loc", seq_loc)
        subset.insert(2, "Matched Site", site)
        subset.insert(3, "Target Mass (Search)", tgt_mass)
        subset.insert(4, "Lower Bound", lower)
        subset.insert(5, "Upper Bound", upper)
        subset["Delta to Target (Da)"] = subset["Mass"] - tgt_mass
        matches.append(subset)

    if not matches:
        return pd.DataFrame()

    df = pd.concat(matches, ignore_index=True)

    # --- Feature engineering ---
    df["Range Width"] = df["Upper Bound"] - df["Lower Bound"]
    df["Seq Length"] = df["Matched Sequence"].astype(str).str.len()
    df["Delta Abs"] = np.abs(df["Delta to Target (Da)"])
    df["Rel Delta %"] = (df["Delta Abs"] / df["Target Mass (Search)"]) * 100
    df["Is_Centre"] = ((df["Mass"] - df["Lower Bound"]) /
                       (df["Range Width"] + 1e-9))  # position in range (0–1)
    df["Is_Centre"] = df["Is_Centre"].clip(0, 1)

    # --- Label: tighter tolerance for high precision ---
    df["Is_Match"] = (df["Delta Abs"] < 0.02).astype(int)
    return df


def train_model(df):
    """Train a tuned Random Forest model for peptide match prediction."""
    features = ["Mass", "Target Mass (Search)", "Range Width",
                "Delta to Target (Da)", "Delta Abs", "Rel Delta %",
                "Seq Length", "Is_Centre"]
    X = df[features]
    y = df["Is_Match"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest pipeline with standardization
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    # Hyperparameter tuning for accuracy
    param_grid = {
        "rf__n_estimators": [100, 300],
        "rf__max_depth": [8, 12, 16],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf": [1, 2],
        "rf__class_weight": ["balanced"]
    }

    search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    print("✅ Best Parameters:", search.best_params_)
    print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = best_model.named_steps["rf"].feature_importances_
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
        "Importance", ascending=False
    )
    print("\n📊 Feature Importances:\n", imp_df.to_string(index=False))

    joblib.dump(best_model, "peptide_match_model_accurate.pkl")
    print("\n💾 Model saved to peptide_match_model_accurate.pkl")
    return best_model


def predict_new(model, new_data):
    """Predict match probability for new entries."""
    preds = model.predict(new_data)
    probs = model.predict_proba(new_data)
    new_data["Predicted_Match"] = preds
    new_data["Match_Probability"] = probs[:, 1]
    return new_data


def main(search_path, results_path):
    df = prepare_dataset(search_path, results_path)
    if df.empty:
        print("No valid data found — check your input files.")
        return

    print(f"Prepared dataset with {len(df)} entries.")
    model = train_model(df)

    # Test on a random sample
    sample = df.sample(5, random_state=1)[
        ["Mass", "Target Mass (Search)", "Range Width", "Delta to Target (Da)",
         "Delta Abs", "Rel Delta %", "Seq Length", "Is_Centre"]
    ]
    results = predict_new(model, sample)
    print("\n🧪 Sample Predictions:\n", results)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python peptide_match_predictor_accurate.py <search.xlsx> <results.xlsx>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
