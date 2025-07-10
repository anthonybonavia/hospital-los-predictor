# los_app2.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1) Features setup
FEATURES = [
    "age", "charlson", "sapsii", "ventilation",
    "bun_mg_dl", "creatinine_mg_dl", "mechanically_ventilated",
    "sofa_score", "respiration", "coagulation", "liver",
    "cardiovascular", "cns", "renal"
]
NUMERIC = [f for f in FEATURES if f not in ["ventilation", "mechanically_ventilated"]]
BINARY = ["mechanically_ventilated"]
CATEGORICAL = ["ventilation"]
TARGET = "outcome"

# 2) Preprocessor: numeric impute, binary impute, categorical impute+one-hot
preprocessor = ColumnTransformer([
    ("num", IterativeImputer(random_state=0), NUMERIC),
    ("bin", SimpleImputer(strategy="most_frequent", fill_value=0), BINARY),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ]), CATEGORICAL),
])

# 3) Multiclass classifier pipeline
clf_pipeline = Pipeline([
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200,
                                   class_weight="balanced",
                                   random_state=0))
])

@st.cache_resource
def load_data():
    # 1) Dropbox “Copy link” → replace dl=0 with dl=1
    url = "https://www.dropbox.com/scl/fi/8o976w9g9k3heeclbe8mb/my_table.csv?rlkey=0114ysndn9aa27d15j2nd36je&st=a7pf2lrs&dl=1"
    
    # 2) Read straight from Dropbox
    df = pd.read_csv(url)
    
    # 3) Map ventilated field
    df["mechanically_ventilated"] = df["mechanically_ventilated"].map({
        "Yes": 1,
        "No": 0,
        "InvasiveVent": 1,
        "SupplementalOxygen": 0,
        np.nan: np.nan
    }).astype(float)
    
    # 4) Fill & cast sepsis3
    df["sepsis3"] = df["sepsis3"].fillna(0).astype(int)
    
    # 5) Define multiclass outcome
    df[TARGET] = np.where(
        df["days_until_death"].notna() & (df["days_until_death"] < 10),
        "early_death",
        np.where(
            df["hospital_los_days"] >= 10,
            "long_los",
            "short_los"
        )
    )
    return df


@st.cache_resource
def train_and_metrics(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25,
        stratify=y, random_state=42
    )
    from sklearn.base import clone
    model = clone(clf_pipeline)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    # dict for dataframe
    report_dict = classification_report(y_te, y_pred,
                                        output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    return report_df, model

# ── App UI ─────────────────────────────────────────────────────

st.title("Multiclass LOS Classifier (Early Death, Short, Long)")

# explain classes
st.markdown("""
**Class definitions**:
- **early_death**: patient died in-hospital before 10 days
- **short_los**: discharged or survived with hospital stay < 10 days
- **long_los**: hospital stay ≥ 10 days
""" )

# load data and split by sepsis
df = load_data()
sep_df = df[df["sepsis3"] == 1]
non_df = df[df["sepsis3"] == 0]

# train and get metrics
report_sep, model_sep = train_and_metrics(sep_df)
report_non, model_non = train_and_metrics(non_df)

# display metrics in tables
st.subheader("Performance: Sepsis-3 Positive Cohort")
st.table(report_sep.style.format({
    "precision":"{:.2f}",
    "recall":"{:.2f}",
    "f1-score":"{:.2f}",
    "support":"{:.0f}"
}))

st.subheader("Performance: Sepsis-3 Negative Cohort")
st.table(report_non.style.format({
    "precision":"{:.2f}",
    "recall":"{:.2f}",
    "f1-score":"{:.2f}",
    "support":"{:.0f}"
}))

# ── Individual prediction ─────────────────────────────────────
st.subheader("Predict an Individual Patient Outcome")
with st.form("patient_form"):
    inputs = {}
    for feat in FEATURES:
        if feat in NUMERIC:
            inputs[feat] = st.number_input(feat.replace("_"," ").title(), value=float("nan"))
        elif feat in BINARY:
            inputs[feat] = st.selectbox(feat.replace("_"," ").title(), ["Unknown","Yes","No"])
        else:  # categorical
            inputs[feat] = st.selectbox(feat.replace("_"," ").title(), ["Unknown"] +
                                 list(df[feat].dropna().unique()))
    seps = st.selectbox("Meets Sepsis-3 criteria?", ["Yes","No"])
    submitted = st.form_submit_button("Compute")

if submitted:
    # build patient df
    pat = {}
    for feat, val in inputs.items():
        if feat in NUMERIC:
            pat[feat] = val
        elif feat == "mechanically_ventilated":
            pat[feat] = 1 if val=="Yes" else 0 if val=="No" else np.nan
        else:
            pat[feat] = val if val!="Unknown" else np.nan
    pat_df = pd.DataFrame([pat])
    model = model_sep if seps=="Yes" else model_non
    pred = model.predict(pat_df)[0]
    probs = model.predict_proba(pat_df)[0]
    st.write(f"**Predicted class:** {pred}")
    st.write("**Class probabilities:**")
    proba_df = pd.DataFrame([probs], columns=model.classes_)
    st.table(proba_df.style.format("{:.2%}"))
