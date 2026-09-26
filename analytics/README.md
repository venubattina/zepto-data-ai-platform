# Module 2 — Analytics Pipeline (/analytics)

## Overview
An end-to-end analytical and machine learning pipeline profiling the Titanic dataset, addressing missingness via threshold-governed rules, diagnosing distributional skewness, and training leakage-free classification and regression models.

## Part A: Profiling, Cleaning, and Data Story
1. **Offline Fallback**: `titanic.csv` committed in `/analytics`. The raw dataset was loaded once via Seaborn, serialized, and reused across all subsequent stages.
2. **Missing-Value Strategy**:
   - `embarked` / `embark_town` (0.22% missing): Below 5% threshold \(\rightarrow\) Dropped affected rows.
   - `age` (19.87% missing): Between 5% and 30% threshold \(\rightarrow\) Imputed with median.
   - `deck` (77.22% missing): Exceeds 30% threshold \(\rightarrow\) Dropped column to prevent artificial noise.
3. **Univariate Analysis & Outliers**:
   - `age`: 11 IQR outliers outside range [-6.69, 64.82].
   - `fare`: 116 IQR outliers outside range [-26.72, 65.63].
   - **Fare Skewness**: Mode (8.05) < Median (14.45) < Mean (32.20). Distribution is strongly right-skewed.
4. **Bivariate Breakdown & Correlations**:
   - Survival Rate by Sex: Female (74.20%) vs. Male (18.89%).
   - Survival Rate by Pclass: 1st (62.96%), 2nd (47.28%), 3rd (24.24%).
   - Top off-diagonal correlations (restricted to `[survived, pclass, age, sibsp, parch, fare]`):
     - `pclass` & `fare` (\(r = -0.549\)): Higher class status correlated with significantly higher ticket fares.
     - `survived` & `pclass` (\(r = -0.338\)): 3rd class passengers experienced substantially lower survival rates.

## Part B: Predictive Modeling Benchmark

### Classification Models
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8034 | 0.7705 | 0.6912 | 0.7287 | 0.8521 |
| **Decision Tree (depth=4)**| 0.8146 | 0.8148 | 0.6471 | 0.7213 | 0.8310 |
| **Random Forest (Tuned)** | **0.8371**| **0.8254**| **0.7353**| **0.7778**| **0.8794**|

### Imbalance Strategy Comparison (Random Forest)
- **Baseline (None)**: F1 = 0.7580, Precision = 0.8100, Recall = 0.7120
- **`class_weight='balanced'`**: F1 = 0.7640, Precision = 0.7850, Recall = 0.7440
- **SMOTE (Train fold only)**: F1 = 0.7710, Precision = 0.8010, Recall = 0.7430
- *Conclusion*: SMOTE applied strictly to the training fold produced the best harmonic balance between precision and recall without test-set leakage.

### Hyperparameter Tuning
- Estimator: `RandomForestClassifier(oob_score=True, random_state=42)`
- Best Parameters: `{'clf__max_depth': 6, 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}`
- **Out-of-Bag (OOB) Score**: 0.8214

### Fare Regression Sub-Task
- Metrics: MAE = 24.12, RMSE = 41.34, \(R^2\) = 0.3842, Adjusted \(R^2\) = 0.3621.
- **Heteroscedasticity Conclusion**: Residual plots exhibit a pronounced widening fan pattern as predicted fare increases, confirming strong heteroscedasticity due to extreme high-fare outliers.

### Production Deployment Recommendation
The tuned **Random Forest Classifier** is selected for production deployment because it yields the highest F1 score (0.7778) and ROC-AUC (0.8794) across all evaluated models, effectively balancing false-positive rates with passenger recall. The entire pipeline (ColumnTransformer + model) is exported to `analytics/best_classifier_pipeline.joblib`.
