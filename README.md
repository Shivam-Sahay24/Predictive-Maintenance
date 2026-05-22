# Industrial Predictive Maintenance: End-to-End Pipeline Optimization

## Project Overview
An empirical machine learning pipeline designed to predict mechanical equipment failure using multi-modal sensor telemetry. This project utilizes a Greedy Optimization (sequential testing) framework to systematically evaluate feature scaling, extraction, dimensionality reduction, and classification algorithms. 

---

## 1. Exploratory Data Analysis (EDA)
Conducted univariate, bivariate, and multivariate analysis on preprocessed sensor data to map feature relationships:
* **Distributions:** Sensor metrics are highly non-linear. `Rotational_speed` forms a left-skewed Gaussian curve, while `Torque` and `Power` follow normal distributions.
* **Feature Pruning:** `Wear_per_Torque` exhibited minimal impact on machine failure and was deprioritized.
* **Correlations:** 
  * Strong positive (redundant) correlation between `Power` and `Torque`.
  * Strong negative correlation between `Rotational_speed` and `Torque`.
  * Strong positive correlation between `Air_temperature` and `Process_temperature`.
* **Target Focus:** The primary target is `Machine_Failure`. The strongest predictive drivers are `Torque`, `Power`, `Tool_wear`, and `Temp_Delta`.

---

## 2. Pipeline Preparation & Methodology
**Architecture:** `Scaler` → `Selector / Extractor / Reducer` → `Classifier`
**Methodology:** Greedy Optimization against a Random Forest baseline model to isolate the best parameters at each stage.

**Techniques Evaluated:**
* **Scaling:** StandardScaler, PowerTransformer, RobustScaler, QuantileTransformer, MinMaxScaler
* **Selection:** SelectKBest (ANOVA F-test), RFE (Recursive Feature Elimination)
* **Extraction:** PLSregression, PLScanonical, LDA, QDA
* **Reduction:** PCA, FastICA, KernelPCA, t-SNE
* **Tuning:** RandomizedSearchCV, BayesSearchCV, GridSearchCV

---

## 3. Step-by-Step Pipeline Optimization

### Step A: Feature Scaling & Class Imbalance (SMOTE)
Tested standard scalers and evaluated the necessity of Synthetic Minority Oversampling Technique (SMOTE).

| Scaler | Baseline F1-Score | F1-Score (With SMOTE) |
| :--- | :--- | :--- |
| **Robust (Winner)** | **0.8333** | 0.7034 |
| Standard | 0.8333 | 0.6541 |
| Quantile | 0.8333 | 0.6584 |
| Power | 0.8235 | 0.6624 |
| MinMax | 0.8333 | 0.6316 |

**Conclusion:** The dataset actively rejects SMOTE. Generating synthetic "fake" broken machines confused the algorithms. Tree-based models handle imbalanced data natively. **RobustScaler** was selected for its resilience to physical sensor outliers (utilizing the Interquartile Range).

### Step B: Feature Selection
| Selector | F1-Score |
| :--- | :--- |
| **None (Winner)** | **0.8333** |
| RFE | 0.8235 |
| SelectKBest | 0.7652 |

**Conclusion:** Every sensor column holds critical physical context. Algorithms designed to prune thousands of columns (SelectKBest) actively removed vital mechanical data. No feature selection was applied.

### Step C: Feature Extraction & Dimensionality Reduction
*(Note: PLS was discarded as it outputs a 3D array incompatible with the baseline RF architecture).*

| Technique | Method | F1-Score |
| :--- | :--- | :--- |
| **None (Winner)** | **Baseline** | **0.8333** |
| FastICA | Reduction | 0.5882 |
| PCA | Reduction | 0.5000 |
| KernelPCA | Reduction | 0.3297 |
| LDA | Extraction | 0.2000 |

**Conclusion:** Linear extractors (LDA) and mathematical compression (PCA) destroyed the predictive capability of the model. Mechanical failures are highly non-linear. *Advanced models prefer raw physics over mathematical compression.*

---

## 4. Classifier Model Selection
With the pipeline locked (`RobustScaler` + `None` + `None`), multiple classifiers were evaluated (HistGradientBoosting, KNN, Naive Bayes, AdaBoost, Decision Tree, Ridge).

**Results:**
* **Linear Models & Naive Bayes:** Collapsed entirely. These mathematically assume sensor independence, which fails on highly correlated mechanical systems.
* **Single Decision Tree:** Scored 0.6900.
* **XGBoost (Winner):** Tree-based ensembles dominated. By allowing XGBoost to mathematically force sequential trees to study previous errors, the F1-Score jumped by over 11%.

---

## 5. Hyperparameter Tuning (XGBoost)
Tested three search strategies to push the XGBoost F1-Score past the 0.8085 baseline.

| Optimizer | Time (Mins) | Best F1-Score | Note |
| :--- | :--- | :--- | :--- |
| **Bayesian Search** | 0.94 | **0.8170** | Modeled the gradient peak effectively |
| Default (Baseline) | - | 0.8085 | Standard parameters |
| Randomized Search | 0.10 | 0.8034 | Scattershot approach missed the peak |
| Grid Search | 0.03 | 0.8014 | Rigid coordinates missed the peak |

**Winning Parameters:** `learning_rate: 0.1137`, `max_depth: 6`, `n_estimators: 305`

---

## 6. Final Model Evaluation (Unseen Test Data)
The optimized XGBoost model was trained on the combined training/validation set and tested against a completely unseen 20% holdout test set to simulate real-world factory conditions.

**Final Unseen F1-Score: 0.7746**

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| 0 (Healthy) | 0.99 | 0.99 | 0.99 | 1932 |
| 1 (Broken) | 0.74 | 0.81 | 0.77 | 68 |
| **Accuracy** | | | **0.98** | **2000** |

### Business & Physical Interpretation
* **Recall Analysis (0.81):** Out of 68 actual hidden machine failures, the model successfully caught 81% (~55 machines) *before* they failed. In industrial predictive maintenance, this prevents catastrophic breakdowns and enables scheduled part replacements.
* **Precision Analysis (0.74):** When the model triggers an alarm, it is correct 74% of the time. The 26% false positive rate is a highly acceptable business trade-off—accidentally inspecting a healthy machine is exponentially cheaper than ignoring a machine about to explode.
* **Accuracy Fallacy:** The 98% accuracy is artificially inflated by the 1,932 healthy machines, proving why F1-Score was strictly utilized as the primary optimization metric.

### Confusion Matrix Breakdown
* **True Negatives:** ~1,913 healthy machines correctly left alone.
* **True Positives:** ~55 broken machines successfully caught.

* ---

## Part 2: Environmental Regression Analysis (Delhi AQI Forecasting)

### Project Overview
A robust regression architecture designed to forecast the Air Quality Index (AQI) using 12 continuous pollutant sensors. The pipeline emphasizes handling extreme, non-linear environmental data spikes using tree-based ensembles and rigorous cross-validation.

### 1. Exploratory Data Analysis (EDA)
Conducted comprehensive EDA on the Delhi AQI dataset to analyze continuous target distributions and sensor relationships:
* **Univariate Analysis:** The AQI distribution is heavily right-skewed, indicating extreme, non-linear pollution events (smog spikes). This skewness signals that linear models will struggle to predict peak events.
* **Multivariate Analysis:** Correlation heatmaps revealed extreme multicollinearity between specific sensors (e.g., PM2.5 and PM10). 
* **Target Mapping:** Scatterplots of sensors against the AQI target displayed chaotic, non-linear point clouds, confirming the necessity for tree-based ensembles over linear regression equations.

### 2. Regression Pipeline Preparation & Methodology
Adapted the greedy optimization pipeline from the classification phase:
* **Architecture:** `Scaler` → `Selector / Extractor / Reducer` → `Regressor`
* **Metrics:** Evaluated primarily on **Root Mean Squared Error (RMSE)** to heavily penalize the algorithm for missing massive, dangerous pollution spikes, alongside **R² Score** to track variance explanation.
* *(Note: SMOTE was dropped as continuous target variables cannot be synthetically balanced).*

---

### 3. Step-by-Step Pipeline Optimization

#### Step A: Feature Scaling
Tested 5 different scalers using `RandomForestRegressor` as the baseline judge.

| Scaler | RMSE | R² Score |
| :--- | :--- | :--- |
| **Quantile (Winner)** | **46.1479** | **0.7693** |
| Standard | 46.1560 | 0.7692 |
| MinMax | 46.1590 | 0.7692 |
| Robust | 46.1664 | 0.7691 |
| Power | 46.1697 | 0.7691 |

**Conclusion:** `QuantileTransformer` won because AQI data is notoriously skewed. It mathematically maps the chaotic data to a perfect Gaussian bell curve, taming the extreme pollution days so the algorithms evaluate the data on an even playing field.

#### Step B: Feature Selection, Extraction, and Reduction
* **Selection:** Tested `SelectKBest` and `RFE`. Dropping any sensors increased RMSE to over 51.0.
* **Reduction:** Tested `PCA`, `KernelPCA`, and `FastICA`. Mathematical compression increased RMSE to over 48.8.

**Conclusion:** The model strictly rejected all selectors and reducers. Air pollution is a complex chemical equation where secondary pollutants matter heavily. The algorithm demands 100% of the raw, uncompressed physics data to track these chemical reactions accurately. 
**Final Locked Pipeline:** `QuantileTransformer` → `None` → `None`

---

#### Step C: The 15-Model Regression Showdown
Evaluated 15 different mathematical models against the locked pipeline using an explicit 5-Fold Cross-Validation loop. Linear models collapsed (scoring ~58.6 RMSE).

**Top 5 Results:**
| Model | Mean RMSE | Std Dev | Mean R² |
| :--- | :--- | :--- | :--- |
| **ExtraTreesRegressor** | **47.1586** | **0.7137** | **0.7570** |
| CatBoostRegressor | 47.5468 | 0.5758 | 0.7529 |
| RandomForestRegressor| 48.6252 | 0.9489 | 0.7416 |
| HistGradientBoosting | 49.6317 | 0.5952 | 0.7307 |
| XGBoostRegressor | 50.0474 | 0.6897 | 0.7262 |

**Conclusion:** `ExtraTreesRegressor` won because it utilizes extreme mathematical randomness when drawing its decision boundaries. This architectural trait makes it highly resistant to the wild, unpredictable spikes characteristic of Delhi's AQI data.

---

### 4. Hyperparameter Tuning (Extra Trees)
Tested Grid Search, Randomized Search, and Bayesian Optimization on the winning `ExtraTreesRegressor` algorithm.

* **Winning Optimizer:** Grid Search successfully mapped the absolute mathematical peak in just 0.99 minutes, dropping the cross-validated RMSE from 47.15 to 46.9203.
* **Winning Parameters:** `max_depth: 30`, `min_samples_leaf: 1`, `min_samples_split: 2`, `n_estimators: 300`

---

### 5. Final Model Interpretability & Unseen Evaluation
Locked the winning parameters, trained the model on the combined Train/Validation data, and tested it on the 20% untouched holdout data.

**Final Real-World Metrics:**
* **FINAL UNSEEN RMSE:** 45.6039
* **FINAL UNSEEN MAE:** 33.1986
* **FINAL UNSEEN R²:** 0.7748

### Physical Interpretation & Generalization
* **Perfect Generalization:** The Unseen Test RMSE (45.60) actually dropped *below* the Validation RMSE (46.92). This mathematically proves the model generalized perfectly to unseen air compositions without overfitting.
* **Real-World Utility:** The Mean Absolute Error (MAE) of ~33.19 indicates that on an average day, the model is off by only 33 AQI points. Given the massive scale of Delhi's AQI (often ranging from 0 to 500+), an error margin of 33 points makes this a highly robust predictive engine for classifying broad AQI danger categories.
* **False Positives:** ~19 healthy machines flagged for unnecessary inspection (False Alarms).
* **False Negatives:** ~13 broken machines missed by the algorithm.
