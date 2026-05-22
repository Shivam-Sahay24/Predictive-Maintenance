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
* **False Positives:** ~19 healthy machines flagged for unnecessary inspection (False Alarms).
* **False Negatives:** ~13 broken machines missed by the algorithm.
