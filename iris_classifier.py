import os
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = OUTPUT_DIR.parent
MPL_CACHE_DIR = WORKSPACE_DIR / "work" / "matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species_id"})
    df["species"] = df["species_id"].map(dict(enumerate(iris.target_names))).str.title()
    return df


def save_exploration_plots(df: pd.DataFrame) -> None:
    feature_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    sns.set_theme(style="whitegrid", context="notebook")

    pair_grid = sns.pairplot(
        df,
        vars=feature_columns,
        hue="species",
        diag_kind="hist",
        palette="Set2",
        plot_kws={"alpha": 0.85, "s": 48, "edgecolor": "white", "linewidth": 0.4},
    )
    pair_grid.fig.suptitle("Iris Feature Relationships by Species", y=1.02)
    pair_grid.savefig(OUTPUT_DIR / "iris_scatter_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(pair_grid.fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, column in zip(axes.ravel(), feature_columns):
        sns.histplot(
            data=df,
            x=column,
            hue="species",
            multiple="stack",
            palette="Set2",
            bins=16,
            edgecolor="white",
            ax=ax,
        )
        ax.set_title(column.title())
        ax.set_xlabel("Measurement")
        ax.set_ylabel("Count")

    fig.suptitle("Iris Measurement Distributions", fontsize=16)
    fig.savefig(OUTPUT_DIR / "iris_histograms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(df: pd.DataFrame) -> str:
    feature_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    x = df[feature_columns]
    y = df["species"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    best_name = None
    best_score = -1.0
    best_predictions = None

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        score = accuracy_score(y_test, predictions)
        results.append((name, score, predictions))
        if score > best_score:
            best_name = name
            best_score = score
            best_predictions = predictions

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Iris Species Classification Report",
        "",
        "Dataset: scikit-learn's copy of the classic UCI Iris dataset.",
        "",
        "## Data",
        "",
        f"- Rows: {len(df)}",
        f"- Features: {', '.join(feature_columns)}",
        f"- Classes: {', '.join(labels)}",
        "- Preprocessing: no missing values; StandardScaler used for Logistic Regression and KNN.",
        f"- Train/test split: 80/20 stratified split with random_state={RANDOM_STATE}.",
        "",
        "## Test Accuracy",
        "",
    ]

    for name, score, _ in results:
        report.append(f"- {name}: {score:.3f}")

    report.extend(
        [
            "",
            f"Best model on this split: **{best_name}** with accuracy **{best_score:.3f}**.",
            "",
            "## Classification Report",
            "",
            "```text",
            classification_report(y_test, best_predictions, labels=labels),
            "```",
            "",
            "## Generated Visuals",
            "",
            "- `iris_scatter_matrix.png`: pairwise feature scatter plots and histograms by species.",
            "- `iris_histograms.png`: feature distributions by species.",
            "- `confusion_matrix.png`: confusion matrix for the best model.",
            "",
        ]
    )

    return "\n".join(report)


def main() -> None:
    df = load_dataset()
    df.to_csv(OUTPUT_DIR / "iris_dataset.csv", index=False)
    save_exploration_plots(df)
    report = train_and_evaluate(df)
    (OUTPUT_DIR / "iris_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
