# Predictive Modeling for Breast Cancer Classification with PCA

A crucial data science pipeline that utilizes **Principal Component Analysis (PCA)** to reduce the feature dimensionality of the Wisconsin Breast Cancer dataset, balancing model complexity with predictive performance for critical medical diagnostics.

---

## Project Title & Short Description

**Title:** Breast Cancer Classification: Feature Reduction with Principal Component Analysis (PCA)

**Description:** This project implements an end-to-end Machine Learning pipeline focusing on **dimensionality reduction**. It applies **PCA** to transform 30 raw features into a smaller set of orthogonal components that retain the majority of the original data's variance (targeted at **≥95%**). The performance of the reduced model is then validated using **Logistic Regression**.

---

## Problem Statement / Goal

The dataset contains 30 highly correlated features, leading to model complexity, increased training time, and potential overfitting. The core objective is to:
1.  **Reduce Feature Space**: Systematically reduce the initial 30 features down to the minimum number of **Principal Components (PCs)** required to explain at least **95%** of the data's total variance.
2.  **Validate Performance**: Confirm that the **Logistic Regression** model trained on the reduced feature set maintains a comparable or superior **Cross-Validation Accuracy** relative to the model trained on the original, full feature set.

---

## Tech Stack / Tools Used

The solution is implemented using the standard Python data science stack, with a focus on Scikit-learn for modeling and PCA:

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Data Handling** | Pandas NumPy | Data manipulation and numerical operations |
| **Preprocessing**| StandardScaler | Normalizing data (a prerequisite for PCA) |
| **Dimensionality**| PCA (Scikit-learn) | Principal Component Analysis for feature transformation |
| **Modeling** | LogisticRegression | Baseline and final classifier for performance comparison |
| **Visualization**| Matplotlib Seaborn | Creating the **Explained Variance Plot** and **PCA Loadings Heatmap** |

---

## Approach / Methodology

1.  **Data Preprocessing**: The dataset is loaded and feature scaling is performed using `StandardScaler` to ensure all features contribute equally to the PCA calculation.
2.  **PCA Application**: PCA is fitted to the scaled data. The cumulative explained variance ratio is calculated to determine the optimal number of Principal Components (PCs).
3.  **Model Comparison**: Two Logistic Regression models are trained and evaluated using 5-fold cross-validation:
    * **Baseline Model**: Trained on the original 30 scaled features.
    * **PCA Model**: Trained on the reduced set of PCs (those retaining ≥95% variance).
4.  **Interpretation**: The Loadings Heatmap is analyzed to identify the original features (e.g., `radius`, `texture`, `perimeter`, `area` families) that primarily contribute to the variance captured by the first few PCs.

---

## Results / Key Findings

* **Dimensionality Reduction**: The project successfully identified that a significantly reduced number of Principal Components can retain **≥95%** of the variance present in the original 30 features.
* **Performance Maintenance**: The cross-validation accuracy of the Logistic Regression model trained on the reduced feature set was comparable to the model trained on the original 30 features.
* **Interpretability**: The 2D PCA plot visually confirmed the clear separation between the **benign** and **malignant** classes in the transformed space, enhancing interpretability for stakeholders.

---

## Topic Tags

MachineLearning PCA DimensionalityReduction FeatureEngineering BreastCancer Classification LogisticRegression Scikit-learn

---

## How to Run the Project

### 1. Install Requirements

Install all necessary packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
