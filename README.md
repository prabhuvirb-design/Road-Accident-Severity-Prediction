# Project 10: Road Accident Severity Prediction

## 1. Project Overview
This project builds an end-to-end machine learning classification pipeline to predict road accident casualty involvement (`Casualty_class`: `Driver or rider`, `Passenger`, `Pedestrian`, or `na`) and identify high-risk environmental and temporal conditions using the **RTA Dataset** (`12,316` records, `25` columns).

---

## 2. Dataset & Preprocessing
- **Source Dataset:** `rta_dataset.csv` (`12,316` rows, `25` columns).
- **Target Variable:** `Casualty_class` (4 classes).
- **Feature Engineering:** Extracted numeric `Accident_hour` (`0`–`23`) from the raw `Time` string column, retaining `24` predictive features in `X`.
- **Missing Value Handling:** Rather than dropping informative columns with moderate missing rates (such as `Defect_of_vehicle` at ~36% and `Service_year_of_vehicle` at ~32%), missing values are handled cleanly inside a leakage-free `ColumnTransformer` pipeline:
  - **Numeric Features:** Median imputation (`SimpleImputer(strategy="median")`) followed by `StandardScaler`.
  - **Categorical Features:** Most-frequent imputation (`SimpleImputer(strategy="most_frequent")`) followed by `OneHotEncoder(handle_unknown="ignore")`.
- **Train-Test Split:** 80% Training (`9,852` rows) / 20% Testing (`2,464` rows) using stratified sampling (`random_state=42`).

---

## 3. Model Experiments & Results (5-Fold Stratified CV)
We evaluated a linear baseline (`Logistic Regression`) alongside two ensemble classifiers (`Random Forest` and `Gradient Boosting`) using 5-fold Stratified Cross-Validation and held-out test metrics:

| Model | CV Macro F1 | Test Accuracy | Test Macro F1 |
| :--- | :---: | :---: | :---: |
| **Random Forest (Best Model)** | **0.5787** | **0.7541** | **0.5763** |
| Logistic Regression (Baseline) | 0.5717 | 0.7289 | 0.5717 |
| Gradient Boosting | 0.4860 | 0.7537 | 0.4860 |

- **Best Model Selection:** **Random Forest** achieved the strongest balance across both cross-validation (`0.5787` Macro F1) and test evaluation (`75.41%` Accuracy, `0.5763` Macro F1).

---

## 4. Key Insights & Risk Analysis
- **Top Predictive Features:** `Sex_of_casualty`, `Accident_hour`, `Number_of_vehicles_involved`, `Age_band_of_casualty`, and `Cause_of_accident` contributed the highest feature importance in the Random Forest model.
- **Confusion Matrix Findings:** The model separates `na` (`889/889` exact) and `Driver or rider` (`812` correct) reliably, while minority classes (`Pedestrian` and `Passenger`) show overlap with `Driver or rider` due to shared road/vehicle attributes.
- **High-Risk Lighting Conditions (`risk_by_light`):**
  - `Darkness - lights unlit` shows the highest pedestrian casualty share (**15.00%**), compared to **13.55%** in `Darkness - lights lit` and **12.23%** in `Daylight`.

---

## 5. Limitations & Future Work
- **Class Imbalance:** Minority classes (`Pedestrian` and `Passenger`) have lower recall than majority classes (`Driver or rider` and `na`).
- **Next Steps:** Apply `class_weight="balanced"` or SMOTE oversampling, perform hyperparameter tuning (`RandomizedSearchCV`), and benchmark gradient-boosted libraries such as LightGBM or XGBoost.

---

## 6. How to Run
1. Upload `rta_dataset.csv` to `/content/rta_dataset.csv` in Google Colab (or place it in your local project directory).
2. Install dependencies if running locally:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open the notebook and click **Runtime → Restart and run all** to reproduce all preprocessing steps, comparison tables, and visualizations.
