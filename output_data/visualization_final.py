import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# =============================
# Basic settings
# =============================
INPUT_CSV = "model_performance_summary.csv"
OUTPUT_SUMMARY_CSV = "model_performance_summary_means.csv"

MODEL_LABEL_MAP = {
    # canonical / old spellings
    "LogisticRegression": "LR",
    "Logistic Regression": "LR",
    "LR": "LR",

    "MLP": "MLP",

    "RandomForest": "RF",
    "Random Forest": "RF",
    "RF": "RF",

    "SVM": "SVM",

    "kNN": "kNN",
    "KNN": "kNN",
    "K-Nearest Neighbors": "kNN",
    "K Nearest Neighbors": "kNN",
}

DESCRIPTOR_LABEL_MAP = {
    "avalon_descriptors.csv": "Avalon",
    "Avalon": "Avalon",

    "maccs_descriptors.csv": "MACCS",
    "MACCS": "MACCS",

    "mordred_descriptors.csv": "Mordred",
    "Mordred": "Mordred",

    "morgan_descriptors.csv": "Morgan",
    "Morgan": "Morgan",

    "rdkit_descriptors.csv": "RDKit",
    "RDKit": "RDKit",
}

MODEL_ORDER = ["LR", "MLP", "RF", "SVM", "kNN"]
DESCRIPTOR_ORDER = ["Avalon", "MACCS", "Mordred", "Morgan", "RDKit"]

METRIC_CONFIG = [
    {
        "column": "Test_Accuracy",
        "title": "Mean Test Accuracy",
        "filename": "mean_test_accuracy_heatmap.png",
        "cmap": "viridis",
    },
    {
        "column": "Test_F1",
        "title": "Mean Test F1 Score",
        "filename": "mean_test_f1score_heatmap.png",
        "cmap": "magma",
    },
]


# =============================
# Utilities
# =============================
def try_load_json(x):
    try:
        if pd.isna(x):
            return x
        return json.loads(x)
    except Exception:
        return x


def normalize_descriptor_name(value):
    if pd.isna(value):
        return value
    value = str(value).strip()
    value = DESCRIPTOR_LABEL_MAP.get(value, value)
    value = value.replace("_descriptors.csv", "")
    return value


def normalize_model_name(value):
    if pd.isna(value):
        return value
    value = str(value).strip()
    return MODEL_LABEL_MAP.get(value, value)


def prettify_metric_name(metric_name):
    return str(metric_name).replace("_", " ")


def save_csv_safely(df, preferred_filename):
    try:
        df.to_csv(preferred_filename, index=False)
        print(f"\nMean summary saved to {preferred_filename}")
        return preferred_filename
    except PermissionError:
        fallback = (
            f"{Path(preferred_filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.to_csv(fallback, index=False)
        print(
            f"\nCould not overwrite {preferred_filename} "
            f"(it may be open in Excel or locked)."
        )
        print(f"Mean summary saved instead to {fallback}")
        return fallback


def apply_plot_style():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.size": 11,
        }
    )


def warn_unknown_labels(df):
    unknown_models = sorted(
        set(df["Model"].dropna().unique()) - set(MODEL_ORDER)
    )
    unknown_descriptors = sorted(
        set(df["Descriptor"].dropna().unique()) - set(DESCRIPTOR_ORDER)
    )

    if unknown_models:
        print("\n[WARNING] Unknown model labels found after normalization:")
        for x in unknown_models:
            print(f"  - {x}")

    if unknown_descriptors:
        print("\n[WARNING] Unknown descriptor labels found after normalization:")
        for x in unknown_descriptors:
            print(f"  - {x}")


def check_duplicate_pairs(df, stage_name="dataframe"):
    dups = df[df.duplicated(subset=["Model", "Descriptor"], keep=False)]
    if not dups.empty:
        print(f"\n[DEBUG] Duplicate (Model, Descriptor) pairs detected in {stage_name}:")
        print(dups.sort_values(["Model", "Descriptor"]).to_string(index=False))


# =============================
# Data loading
# =============================
def load_and_prepare_data(input_csv):
    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv)

    print("\nUnique Model values before mapping:")
    print(sorted(df["Model"].astype(str).unique()))

    print("\nUnique Descriptor values before mapping:")
    print(sorted(df["Descriptor"].astype(str).unique()))

    if "Best_Params" in df.columns:
        df["Best_Params"] = df["Best_Params"].apply(try_load_json)

    score_columns = ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1"]
    for col in score_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Model"] = df["Model"].apply(normalize_model_name)
    df["Descriptor"] = df["Descriptor"].apply(normalize_descriptor_name)

    print("\nUnique Model values after mapping:")
    print(sorted(df["Model"].astype(str).unique()))

    print("\nUnique Descriptor values after mapping:")
    print(sorted(df["Descriptor"].astype(str).unique()))

    warn_unknown_labels(df)

    print("\nSample of loaded data:")
    print(df.head())

    return df


# =============================
# Summary table
# =============================
def make_summary_table(df):
    print("\nAggregating mean scores by Model and Descriptor ...")
    summary = (
        df.groupby(["Model", "Descriptor"], as_index=False, dropna=False)
        .agg(
            {
                "Test_Accuracy": "mean",
                "Test_Precision": "mean",
                "Test_Recall": "mean",
                "Test_F1": "mean",
            }
        )
    )

    print("\nSummary table before categorical conversion:")
    print(summary)

    known_mask = summary["Model"].isin(MODEL_ORDER) & summary["Descriptor"].isin(DESCRIPTOR_ORDER)
    dropped = summary.loc[~known_mask].copy()
    if not dropped.empty:
        print("\n[WARNING] Dropping rows with unknown Model/Descriptor labels:")
        print(dropped.to_string(index=False))

    summary = summary.loc[known_mask].copy()

    check_duplicate_pairs(summary, stage_name="summary before categorical conversion")

    summary["Model"] = pd.Categorical(summary["Model"], categories=MODEL_ORDER, ordered=True)
    summary["Descriptor"] = pd.Categorical(
        summary["Descriptor"], categories=DESCRIPTOR_ORDER, ordered=True
    )
    summary = summary.sort_values(["Model", "Descriptor"]).reset_index(drop=True)

    print("\nSummary table:")
    print(summary)

    return summary


# =============================
# Visualization
# =============================
def plot_and_save_heatmap(data, value_col, title, cmap, filename, vmin=None, vmax=None):
    check_duplicate_pairs(data, stage_name=f"plot input for {value_col}")

    pivot_table = data.pivot_table(
        index="Model",
        columns="Descriptor",
        values=value_col,
        aggfunc="mean",
    )
    pivot_table = pivot_table.reindex(index=MODEL_ORDER, columns=DESCRIPTOR_ORDER)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.6,
        linecolor="white",
        cbar=True,
        square=False,
        annot_kws={"size": 11},
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    ax.set_title(title, fontsize=16, pad=14, weight="bold")
    ax.set_xlabel("Descriptor", fontsize=14, labelpad=12, weight="bold")
    ax.set_ylabel("Model", fontsize=14, labelpad=12, weight="bold")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=11)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    colorbar_label = prettify_metric_name(value_col)
    cbar.set_label(colorbar_label, fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved heatmap to {filename}")
    plt.show()
    plt.close(fig)


def make_all_heatmaps(summary):
    accuracy_min = summary["Test_Accuracy"].min()
    accuracy_max = summary["Test_Accuracy"].max()
    f1_min = summary["Test_F1"].min()
    f1_max = summary["Test_F1"].max()

    for config in METRIC_CONFIG:
        print(f"\nCreating and saving {config['title']} heatmap ...")
        if config["column"] == "Test_Accuracy":
            vmin, vmax = accuracy_min, accuracy_max
        elif config["column"] == "Test_F1":
            vmin, vmax = f1_min, f1_max
        else:
            vmin, vmax = None, None

        plot_and_save_heatmap(
            summary,
            value_col=config["column"],
            title=config["title"],
            cmap=config["cmap"],
            filename=config["filename"],
            vmin=vmin,
            vmax=vmax,
        )


# =============================
# Main
# =============================
def main():
    apply_plot_style()

    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"{INPUT_CSV} was not found in the current directory.")

    df = load_and_prepare_data(INPUT_CSV)
    summary = make_summary_table(df)

    save_csv_safely(summary, OUTPUT_SUMMARY_CSV)
    make_all_heatmaps(summary)


if __name__ == "__main__":
    main()