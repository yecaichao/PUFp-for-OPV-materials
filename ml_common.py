from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_feature_target(
    feature_csv="output_n/number.csv",
    target_csv="OPV_exp_data.csv",
    id_column="No.",
    target_column="PCE_max(%)",
):
    feature_path = Path(feature_csv)
    target_path = Path(target_csv)
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature csv: {feature_csv}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target csv: {target_csv}")

    feature_df = pd.read_csv(feature_path, index_col=0)
    target_df = pd.read_csv(target_path)

    if id_column not in target_df.columns:
        raise KeyError(f"Missing id column in target csv: {id_column}")
    if target_column not in target_df.columns:
        raise KeyError(f"Missing target column in target csv: {target_column}")

    feature_df.index = feature_df.index.map(str)
    target_df[id_column] = target_df[id_column].map(str)

    merged = feature_df.merge(
        target_df[[id_column, target_column]],
        left_index=True,
        right_on=id_column,
        how="inner",
    )
    if merged.empty:
        raise ValueError("No shared sample ids were found between the feature table and target table.")

    X = merged[feature_df.columns].to_numpy(dtype=float)
    y = merged[target_column].to_numpy(dtype=float)
    ids = merged[id_column].tolist()
    return X, y, ids, feature_df.columns.tolist()


def split_dataset(X, y, test_size=0.2, random_state=111):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def regression_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "CC": np.corrcoef(y_true, y_pred)[0, 1],
    }


def print_metrics(label, metrics):
    print(
        f"{label} -> "
        f"MSE: {metrics['MSE']:.6f}  "
        f"RMSE: {metrics['RMSE']:.6f}  "
        f"MAE: {metrics['MAE']:.6f}  "
        f"R2: {metrics['R2']:.6f}  "
        f"CC: {metrics['CC']:.6f}"
    )


def plot_predictions(y_train, y_pred_train, y_test, y_pred_test, ylabel, output_path=None):
    plt.figure(figsize=(6, 6))
    plt.plot(y_train, y_pred_train, "o", label="Train")
    plt.plot(y_test, y_pred_test, "o", label="Test")
    x = np.linspace(-2, 22, 1000)
    y = np.linspace(-2, 22, 1000)
    plt.plot(x, y, "_", linewidth=2)
    plt.xlim(-2, 22)
    plt.ylim(-2, 22)
    plt.axhline(y=0, lw=1, ls="--", color="r")
    plt.axvline(x=0, lw=1, ls="--", color="r")
    plt.legend()
    plt.xlabel("Experiment PCE%")
    plt.ylabel(ylabel)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
