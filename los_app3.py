import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

FEATURES = [
    "age", "charlson", "sapsii", "ventilation",
    "bun_mg_dl", "creatinine_mg_dl", "mechanically_ventilated",
    "sofa_score", "respiration", "coagulation",
    "liver", "cardiovascular", "cns", "renal"
]
NUMERIC = [f for f in FEATURES if f not in ["ventilation", "mechanically_ventilated"]]
BINARY = ["mechanically_ventilated"]
CATEGORICAL = ["ventilation"]
TARGET = "outcome"

@st.cache_resource
def load_data():
    url = ("https://www.dropbox.com/scl/fi/8o976w9g9k3heeclbe8mb/"
           "my_table.csv?rlkey=0114ysndn9aa27d15j2nd36je&dl=1")
    df = pd.read_csv(url)
    df["mechanically_ventilated"] = df["mechanically_ventilated"].map({
        "Yes": 1, "No": 0, "InvasiveVent": 1, "SupplementalOxygen": 0, np.nan: np.nan
    }).astype(float)
    df["sepsis3"] = df["sepsis3"].fillna(0).astype(int)
    df[TARGET] = np.where(
        df["days_until_death"].notna() & (df["days_until_death"] < 10),
        "early_death",
        np.where(df["hospital_los_days"] >= 10, "long_los", "short_los")
    )
    return df

def prepare_validation(path):
    df = pd.read_csv(path, low_memory=False)
    df["mechanically_ventilated"] = df["mechanically_ventilated"].map({
        "Yes": 1, "No": 0, "InvasiveVent": 1, "SupplementalOxygen": 0, np.nan: np.nan
    }).astype(float)
    df["sepsis3"] = df.get("sepsis3", 0).fillna(0).astype(int)
    df[TARGET] = np.where(
        df["days_until_death"].notna() & (df["days_until_death"] < 10),
        "early_death",
        np.where(df["hospital_los_days"] >= 10, "long_los", "short_los")
    )
    return df

@st.cache_resource
def train_and_metrics(df, model_name):
    model_path = f"{model_name}.pkl"
    report_path = f"{model_name}_report.pkl"

    if os.path.exists(model_path) and os.path.exists(report_path):
        model = joblib.load(model_path)
        report_df = joblib.load(report_path)
    else:
        X = df[FEATURES]
        y = df[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        from sklearn.base import clone
        model = clone(clf_pipeline)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        report_df = pd.DataFrame(classification_report(y_te, y_pred, output_dict=True)).T
        joblib.dump(model, model_path)
        joblib.dump(report_df, report_path)

    return report_df, model

def calibrate_models(model_sep, model_non):
    seps_val = prepare_validation("validate_sepsis.csv")
    seps_val = seps_val[seps_val["sepsis3"] == 1]
    X_seps = seps_val[FEATURES]
    y_seps = seps_val[TARGET]
    cal_sep = CalibratedClassifierCV(estimator=model_sep, cv="prefit", method="isotonic")
    cal_sep.fit(X_seps, y_seps)

    cins_val = prepare_validation("validate_cins.csv")
    cins_val = cins_val[cins_val["sepsis3"] == 0]
    X_cins = cins_val[FEATURES]
    y_cins = cins_val[TARGET]
    cal_non = CalibratedClassifierCV(estimator=model_non, cv="prefit", method="isotonic")
    cal_non.fit(X_cins, y_cins)

    return cal_sep, cal_non

preprocessor = ColumnTransformer([
    ("num", IterativeImputer(random_state=0), NUMERIC),
    ("bin", SimpleImputer(strategy="most_frequent", fill_value=0), BINARY),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ]), CATEGORICAL),
])

clf_pipeline = Pipeline([
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0))
])

st.title("Multiclass LOS Classifier (Early Death, Short, Long)")

st.markdown("""
**Class definitions**:
- **early_death**: died in hospital < 10 days  
- **short_los**: stay < 10 days  
- **long_los**: stay ≥ 10 days  
""")

df = load_data()
sep_df = df[df["sepsis3"] == 1]
non_df = df[df["sepsis3"] == 0]

report_sep, model_sep = train_and_metrics(sep_df, "model_sepsis")
report_non, model_non = train_and_metrics(non_df, "model_nonsepsis")

model_sep, model_non = calibrate_models(model_sep, model_non)

st.subheader("Performance: Sepsis-3 Positive (Internal CV)")
st.table(report_sep.style.format({
    "precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"
}))

st.subheader("Performance: Sepsis-3 Negative (Internal CV)")
st.table(report_non.style.format({
    "precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"
}))

st.subheader("Predict an Individual Patient Outcome")
with st.form("patient_form"):
    inputs = {}
    for feat in FEATURES:
        if feat in NUMERIC:
            inputs[feat] = st.number_input(feat.replace("_", " ").title(), value=float("nan"))
        elif feat in BINARY:
            inputs[feat] = st.selectbox(feat.replace("_", " ").title(), ["Unknown", "Yes", "No"])
        else:
            inputs[feat] = st.selectbox(feat.replace("_", " ").title(), ["Unknown"] + list(df[feat].dropna().unique()))
    seps = st.selectbox("Meets Sepsis-3 criteria?", ["Yes", "No"])
    submitted = st.form_submit_button("Compute")

if submitted:
    pat = {}
    for feat, val in inputs.items():
        if feat in NUMERIC:
            pat[feat] = val
        elif feat in BINARY:
            pat[feat] = 1 if val == "Yes" else 0 if val == "No" else np.nan
        else:
            pat[feat] = val if val != "Unknown" else np.nan

    pat_df = pd.DataFrame([pat])
    model = model_sep if seps == "Yes" else model_non

    pred = model.predict(pat_df)[0]
    probs = model.predict_proba(pat_df)[0]

    st.write(f"**Predicted class:** {pred}")
    st.write("**Class probabilities:**")
    proba_df = pd.DataFrame([probs], columns=model.classes_)
    st.table(proba_df.style.format("{:.2%}"))