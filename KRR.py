import argparse
import itertools

import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, laplacian_kernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ml_common import load_feature_target, split_dataset, regression_metrics, print_metrics, plot_predictions


def make_kernel(X1, X2, kernel, gamma):
    if kernel == "linear":
        return linear_kernel(X1, X2)
    if kernel == "rbf":
        return rbf_kernel(X1, X2, gamma=gamma)
    if kernel == "laplacian":
        return laplacian_kernel(X1, X2, gamma=gamma)
    raise ValueError(f"Unsupported kernel: {kernel}")


def fit_predict_kernel_ridge(X_train, y_train, X_eval, kernel, gamma, alpha):
    K_train = make_kernel(X_train, X_train, kernel, gamma)
    K_train = K_train + alpha * np.eye(K_train.shape[0])
    dual_coef = np.linalg.solve(K_train, y_train)
    K_eval = make_kernel(X_eval, X_train, kernel, gamma)
    return K_eval @ dual_coef


def search_params(X_train, y_train, kernels, gammas, alphas, cv, random_state):
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    best_score = -np.inf
    best_params = None

    for kernel, gamma, alpha in itertools.product(kernels, gammas, alphas):
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            scaler = StandardScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            y_pred = fit_predict_kernel_ridge(
                X_fold_train_scaled, y_fold_train, X_fold_val_scaled, kernel, gamma, alpha
            )
            scores.append(r2_score(y_fold_val, y_pred))

        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_params = {"kernel": kernel, "gamma": gamma, "alpha": alpha}

    return best_params


def main(
    feature_csv="output_n/number.csv",
    target_csv="OPV_exp_data.csv",
    id_column="No.",
    target_column="PCE_max(%)",
    test_size=0.2,
    random_state=111,
    cv=5,
    n_jobs=-1,
    output_plot=None,
    quick=False,
):
    X, y, ids, feature_names = load_feature_target(feature_csv, target_csv, id_column, target_column)
    print("Feature columns:", len(feature_names))
    print("Matched samples:", len(y))

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=test_size, random_state=random_state)
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    if quick:
        kernels = ["rbf"]
        gammas = [1e-4, 1e-3]
        alphas = [1e-4, 1e-2]
    else:
        kernels = ["rbf", "laplacian", "linear"]
        gammas = list((1 / 2 ** np.arange(5, 18, 1.0)) ** 2)
        alphas = list(2 ** np.arange(-40, 0, 1.0))

    best_params = search_params(X_train, y_train, kernels, gammas, alphas, cv, random_state)
    print("Best params:", best_params)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_train = fit_predict_kernel_ridge(
        X_train_scaled,
        y_train,
        X_train_scaled,
        best_params["kernel"],
        best_params["gamma"],
        best_params["alpha"],
    )
    y_pred_test = fit_predict_kernel_ridge(
        X_train_scaled,
        y_train,
        X_test_scaled,
        best_params["kernel"],
        best_params["gamma"],
        best_params["alpha"],
    )

    train_metrics = regression_metrics(y_train, y_pred_train)
    test_metrics = regression_metrics(y_test, y_pred_test)
    print_metrics("Train", train_metrics)
    print_metrics("Test", test_metrics)

    plot_predictions(y_train, y_pred_train, y_test, y_pred_test, "KRR predicted PCE%", output_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Kernel Ridge regressor on OPV PUFp features.")
    parser.add_argument("--feature-csv", default="output_n/number.csv")
    parser.add_argument("--target-csv", default="OPV_exp_data.csv")
    parser.add_argument("--id-column", default="No.")
    parser.add_argument("--target-column", default="PCE_max(%)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=111)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(
        feature_csv=args.feature_csv,
        target_csv=args.target_csv,
        id_column=args.id_column,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        cv=args.cv,
        n_jobs=args.n_jobs,
        output_plot=args.output_plot,
        quick=args.quick,
    )
