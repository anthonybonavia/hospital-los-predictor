# los_app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer   # ← this line
from sklearn.impute      import IterativeImputer, SimpleImputer
from sklearn.compose     import ColumnTransformer
from sklearn.pipeline    import Pipeline
from sklearn.ensemble    import RandomForestRegressor, RandomForestClassifier


# 1) define which columns are numeric vs. binary
NUMERIC = [
    "bun_mg_dl","creatinine_mg_dl","sofa_score",
    "respiration","coagulation","liver",
    "cardiovascular","cns","renal"
]
BINARY = ["mechanically_ventilated","sepsis3"]

# 2) build a preprocessor that imputes missing values
preprocessor = ColumnTransformer([
    ("num", IterativeImputer(random_state=0), NUMERIC),
    ("bin", SimpleImputer(strategy="most_frequent", fill_value=0), BINARY),
])

# 3) wrap regressor + preprocessor into one pipeline
reg_pipeline = Pipeline([
    ("preproc", preprocessor),
    ("reg", RandomForestRegressor(n_estimators=200, random_state=0))
])

clf_pipeline = Pipeline([
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=0
    ))
])

@st.cache_resource
def load_and_train():
    # load your data
    df = pd.read_csv("hospital_los_20250709.csv").dropna(subset=["hospital_los_days"])
    # map booleans→0/1 (leave np.nan where missing)
    df["mechanically_ventilated"] = df["mechanically_ventilated"].map({True:1,False:0})
    df["sepsis3"]                 = df["sepsis3"].map({True:1,False:0})
    # drop rows where we don't even know LOS
    df = df.dropna(subset=["hospital_los_days"])
    df["los_gt_10"] = (df["hospital_los_days"] > 10).astype(int)

    X = df[NUMERIC + BINARY]
    y_reg = df["hospital_los_days"]
    y_clf = df["los_gt_10"]

    # train/test split
    X_train, X_test, y_tr_reg, y_te_reg, y_tr_clf, y_te_clf = st.session_state.splitter

    # fit pipelines
    reg_pipeline.fit(X_train, y_tr_reg)
    clf_pipeline.fit(X_train, y_tr_clf)

    # cache the trained pipelines and test splits for metrics
    return reg_pipeline, clf_pipeline, (X_test,y_te_reg,y_te_clf)

# once at top-level, split data and stash it so load_and_train can reuse it:
if "splitter" not in st.session_state:
    df0 = pd.read_csv("hospital_los_20250709.csv").dropna(subset=["hospital_los_days"])
    df0["mechanically_ventilated"] = df0["mechanically_ventilated"].map({True:1,False:0})
    df0["sepsis3"]                 = df0["sepsis3"].map({True:1,False:0})
    df0["los_gt_10"] = (df0["hospital_los_days"] > 10).astype(int)
    from sklearn.model_selection import train_test_split
    X0 = df0[NUMERIC+BINARY]
    y0_reg = df0["hospital_los_days"]
    y0_clf = df0["los_gt_10"]
    st.session_state.splitter = train_test_split(
        X0, y0_reg, y0_clf,
        stratify=y0_clf,
        test_size=0.25,
        random_state=42
    )

# train (or load from cache)
regressor, classifier, (X_test, y_test_reg, y_test_clf) = load_and_train()

# ------------------------------------------------------
# show metrics
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    roc_auc_score, accuracy_score, f1_score
)
y_pred_reg = regressor.predict(X_test)
y_prob_clf = classifier.predict_proba(X_test)[:,1]
y_pred_clf = (y_prob_clf >= 0.5).astype(int)

st.write("## Model Performance")
st.write(f"Regression → MAE: {mean_absolute_error(y_test_reg,y_pred_reg):.2f}, "
         f"R²: {r2_score(y_test_reg,y_pred_reg):.2f}")
st.write(f"Classification → AUC: {roc_auc_score(y_test_clf,y_prob_clf):.3f}, "
         f"Acc: {accuracy_score(y_test_clf,y_pred_clf):.3f}, "
         f"F1: {f1_score(y_test_clf,y_pred_clf):.3f}")

# ------------------------------------------------------
# interactive prediction form
st.write("## Predict an Individual Patient’s LOS")
with st.form("patient_form"):
    bun   = st.number_input("BUN (mg/dL)", value=float("nan"))
    creat = st.number_input("Creatinine (mg/dL)", value=float("nan"))
    vent  = st.selectbox("Mechanically ventilated?", ["Unknown","Yes","No"])
    sofa  = st.number_input("SOFA score", value=float("nan"))
    resp  = st.number_input("Respiration score", value=float("nan"))
    coag  = st.number_input("Coagulation score", value=float("nan"))
    liv   = st.number_input("Liver score", value=float("nan"))
    card  = st.number_input("Cardiovascular score", value=float("nan"))
    cns   = st.number_input("CNS score", value=float("nan"))
    ren   = st.number_input("Renal score", value=float("nan"))
    seps  = st.selectbox("Meets Sepsis-3 criteria?", ["Unknown","Yes","No"])
    submitted = st.form_submit_button("Compute")

if submitted:
    # map back to numeric + np.nan
    patient = {
        "bun_mg_dl": bun,
        "creatinine_mg_dl": creat,
        "mechanically_ventilated":
            1 if vent=="Yes" else 0 if vent=="No" else np.nan,
        "sofa_score": sofa,
        "respiration": resp,
        "coagulation": coag,
        "liver": liv,
        "cardiovascular": card,
        "cns": cns,
        "renal": ren,
        "sepsis3":
            1 if seps=="Yes" else 0 if seps=="No" else np.nan,
    }
    df_pat = pd.DataFrame([patient])
    # now pipeline handles imputation + prediction
    los_pred = regressor.predict(df_pat)[0]
    p_gt10   = classifier.predict_proba(df_pat)[0,1]

    st.write(f"**Predicted LOS:** {los_pred:.1f} days")
    st.write(f"**P(LOS > 10 days):** {p_gt10:.0%}")
