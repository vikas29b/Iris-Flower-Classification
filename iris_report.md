# Iris Species Classification Report

Dataset: scikit-learn's copy of the classic UCI Iris dataset.

## Data

- Rows: 150
- Features: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
- Classes: Setosa, Versicolor, Virginica
- Preprocessing: no missing values; StandardScaler used for Logistic Regression and KNN.
- Train/test split: 80/20 stratified split with random_state=42.

## Test Accuracy

- Logistic Regression: 0.933
- K-Nearest Neighbors: 0.933
- Decision Tree: 0.967

Best model on this split: **Decision Tree** with accuracy **0.967**.

## Classification Report

```text
              precision    recall  f1-score   support

      Setosa       1.00      1.00      1.00        10
  Versicolor       1.00      0.90      0.95        10
   Virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30

```

## Generated Visuals

- `iris_scatter_matrix.png`: pairwise feature scatter plots and histograms by species.
- `iris_histograms.png`: feature distributions by species.
- `confusion_matrix.png`: confusion matrix for the best model.
