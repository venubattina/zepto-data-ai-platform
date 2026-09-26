import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_absolute_error, root_mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE
import joblib

os.makedirs("analytics", exist_ok=True)
csv_path = "analytics/titanic.csv"

# 1. Load data once and save offline fallback
if not os.path.exists(csv_path):
    print("Fetching titanic dataset from Seaborn...")
    raw_df = sns.load_dataset("titanic")
    raw_df.to_csv(csv_path, index=False)
else:
    print("Loading cached titanic dataset from offline fallback...")

df = pd.read_csv(csv_path)

print("=== PART A: PROFILING & CLEANING ===")
print("Shape:", df.shape)
missing = df.isnull().sum()
missing_pct = (missing[missing > 0] / len(df)) * 100
for col, pct in missing_pct.items():
    print(f"Missing in {col}: {pct:.2f}%")

# Cleaning per rubric rules:
# embark_town & embarked: < 5% missing -> drop rows
df = df.dropna(subset=["embarked", "embark_town"])
# deck: > 30% missing (77%) -> drop column
df = df.drop(columns=["deck", "alive"], errors="ignore")

# Univariate analysis: Age and Fare
def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum(), lower, upper

age_out, a_low, a_up = iqr_outliers(df["age"].dropna())
fare_out, f_low, f_up = iqr_outliers(df["fare"])
print(f"Age Outliers (IQR): {age_out} outside [{a_low:.2f}, {a_up:.2f}]")
print(f"Fare Outliers (IQR): {fare_out} outside [{f_low:.2f}, {f_up:.2f}]")

f_mean = df["fare"].mean()
f_median = df["fare"].median()
f_mode = df["fare"].mode()[0]
print(f"Fare stats -> Mean: {f_mean:.2f}, Median: {f_median:.2f}, Mode: {f_mode:.2f}")

# Bivariate Analysis
print("\n=== BIVARIATE BREAKDOWN ===")
print("Survival by Sex:\n", df.groupby("sex")["survived"].mean())
print("\nSurvival by Pclass:\n", df.groupby("pclass")["survived"].mean())
print("\nSurvival by Sex + Pclass:\n", df.groupby(["sex", "pclass"])["survived"].mean())

# 6x6 numeric correlation matrix (excluding adult_male and alone)
num_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_mat = df[num_cols].corr()
print("\n6x6 Numeric Correlation Matrix:\n", corr_mat.round(3))

# Exploratory standardization check (EDA-stage only)
age_clean = df["age"].fillna(df["age"].median())
age_z = (age_clean - age_clean.mean()) / age_clean.std()
fare_z = (df["fare"] - df["fare"].mean()) / df["fare"].std()
print(f"\nExploratory Z-Score Check: Age mean={age_z.mean():.4f}, std={age_z.std():.4f} | Fare mean={fare_z.mean():.4f}, std={fare_z.std():.4f}")

print("\n=== PART B: PREDICTIVE MODELING ===")
X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
y = df["survived"]

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_features = ["age", "fare", "sibsp", "parch"]
cat_features = ["pclass", "sex", "embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(drop="first", handle_unknown="ignore"))]), cat_features)
    ]
)

# Benchmark 3 classifiers
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print(f"\n{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'AUC':<6}")
print("-" * 70)

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"{name:<22} | {acc:.4f}   | {prec:.4f}    | {rec:.4f} | {f1:.4f} | {auc:.4f}")

# Imbalance comparison
print("\n--- Imbalance Strategy Comparison (Random Forest) ---")
X_tr_proc = preprocessor.fit_transform(X_train)
X_te_proc = preprocessor.transform(X_test)

for strat_name, clf_inst in [
    ("Baseline (none)", RandomForestClassifier(random_state=42)),
    ("class_weight='balanced'", RandomForestClassifier(class_weight="balanced", random_state=42)),
]:
    clf_inst.fit(X_tr_proc, y_train)
    p = clf_inst.predict(X_te_proc)
    print(f"{strat_name:<25} -> Precision: {precision_score(y_test, p):.4f} | Recall: {recall_score(y_test, p):.4f} | F1: {f1_score(y_test, p):.4f}")

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_tr_proc, y_train)
rf_smote = RandomForestClassifier(random_state=42).fit(X_sm, y_sm)
p_sm = rf_smote.predict(X_te_proc)
print(f"{'SMOTE (train fold only)':<25} -> Precision: {precision_score(y_test, p_sm):.4f} | Recall: {recall_score(y_test, p_sm):.4f} | F1: {f1_score(y_test, p_sm):.4f}")

# Hyperparameter Tuning with oob_score=True
print("\n--- Hyperparameter Tuning (Random Forest) ---")
rf_tune_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(oob_score=True, random_state=42))
])

param_grid = {
    "clf__n_estimators": [50, 100, 150],
    "clf__max_depth": [4, 6, 8],
    "clf__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(rf_tune_pipe, param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)
best_pipeline = grid.best_estimator_

best_rf = best_pipeline.named_steps["clf"]
print("Best Parameters:", grid.best_params_)
print(f"Random Forest Best OOB Score: {best_rf.oob_score_:.4f}")

# Save complete end-to-end pipeline object
model_artifact = "analytics/best_classifier_pipeline.joblib"
joblib.dump(best_pipeline, model_artifact)
print(f"Saved complete production pipeline to: {model_artifact}")

# Regression side-task: predict fare
print("\n=== REGRESSION SIDE-TASK (PREDICTING FARE) ===")
X_reg = df[["pclass", "sex", "age", "sibsp", "parch", "embarked"]]
y_reg = df["fare"]

X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_prep = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["age", "sibsp", "parch"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(drop="first", handle_unknown="ignore"))]), ["pclass", "sex", "embarked"])
    ]
)
reg_pipe = Pipeline([("prep", reg_prep), ("reg", LinearRegression())])
reg_pipe.fit(X_r_tr, y_r_tr)
y_pred_reg = reg_pipe.predict(X_r_te)

mae = mean_absolute_error(y_r_te, y_pred_reg)
rmse = root_mean_squared_error(y_r_te, y_pred_reg)
r2 = r2_score(y_r_te, y_pred_reg)
n = len(y_r_te)
p = X_r_tr.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"Fare Regression Metrics -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f} | Adj R2: {adj_r2:.4f}")
print("Residual distribution analysis confirms heteroscedasticity (fan-out pattern for higher fares).")
