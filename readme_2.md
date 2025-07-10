# LOS App 2

This repository contains `los_app2.py`, a Streamlit application for predicting in‑hospital length of stay (LOS) outcomes—including early death, short stay, and long stay—using patient clinical features. It also includes `requirements2.txt` listing the necessary Python dependencies.

---

## 📋 Prerequisites

- **Python**: 3.8+ (recommend 3.9 or newer)
- **Git**: to clone this repository
- **Streamlit**: for running the app

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create a virtual environment** (optional but recommended)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .\.venv\Scripts\activate   # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements2.txt

4. **Create a `.gitignore` file** in the root of your repo with the following contents:

   ```gitignore
   # Python
   __pycache__/
   *.py[cod]
   .venv/
   env/
   venv/
   
   # Streamlit
   .streamlit/
   
   # Data
   *.csv
   ```

   This ensures that temporary files, your virtual environment, and large data files are not committed.
   ```

---

## 🚀 Running the App Locally

With the virtual environment activated and dependencies installed:

```bash
streamlit run los_app2.py
```

This will launch a browser tab showing the interactive web app.

---

## ☁️ Data Loading

The app loads a large CSV (`my_table.csv`) directly from a public URL (e.g. Dropbox). If you need to host your own copy:

1. Upload `my_table.csv` to Dropbox (or another file host).
2. Obtain a **direct download** link ending in `?dl=1`.
3. In `los_app2.py`, update the `url` variable inside `load_data()`:

```python
url = "https://www.dropbox.com/s/<FILE_ID>/my_table.csv?dl=1"
```

---

## 🧠 Model Logic Overview

### 1. Feature Setup

- **FEATURES**: 14 variables including demographics (`age`, `charlson`, `sapsii`), lab values (`bun_mg_dl`, `creatinine_mg_dl`), and organ‑system scores (`sofa_score`, `respiration`, etc.).
- **BINARY**: `mechanically_ventilated` (0/1)
- **CATEGORICAL**: `ventilation` (e.g. `InvasiveVent`, `SupplementalOxygen`)

### 2. Outcome Classes

Built three mutually exclusive classes in `load_data()`:

- ``: patient died in-hospital before 10 days (based on `days_until_death`).
- ``: discharged or survived with LOS < 10 days.
- ``: LOS ≥ 10 days.

### 3. Preprocessing Pipeline

- **Numeric**: Iterative imputation for missing lab and score values.
- **Binary**: Simple imputation (most frequent) for missing 0/1 flags.
- **Categorical**: Fill missing with `Unknown`, then one-hot encode.

### 4. Model Training & Evaluation

- Split each **Sepsis-3+** and **Sepsis-3–** cohort separately (stratified by `outcome`).
- Train a **RandomForestClassifier** (200 trees, balanced classes) in a `Pipeline`.
- Evaluate on a 25% test set and show a **classification report** (precision, recall, F1, support).

### 5. Individual Prediction

- Interactive form to collect **all FEATURES + Sepsis-3 flag**.
- Impute/encode inputs via the same pipeline.
- Select the appropriate cohort model and output:
  - **Predicted class** (`early_death`, `short_los`, or `long_los`).
  - **Class probabilities** for all three outcomes.

---

## ⚙️ Deployment Notes

- Ensure `requirements2.txt` is in the repo root alongside `los_app2.py`.
- Keep your CSV hosted externally (e.g. Dropbox), and update the URL in `load_data()`.
- When deploying to Streamlit Community Cloud or another platform, add any necessary environment variables or config in `.streamlit/config.toml`.

---

## 📄 License & Attribution

Your project’s license details here.

---

*Generated on YYYY-MM-DD*

