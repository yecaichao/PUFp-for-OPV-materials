import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from ml_common import load_feature_target, split_dataset, regression_metrics, print_metrics, plot_predictions


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
        tuned_parameters = [{
            "max_depth": [6, 10],
            "max_features": ["sqrt"],
            "n_estimators": [50, 100],
        }]
    else:
        param_range = [2, 6, 8, 10, 20, 30, 40, 50]
        tuned_parameters = [{
            "max_depth": param_range,
            "max_features": ["auto", "sqrt", "log2"],
            "n_estimators": list(range(5, 200, 5)),
        }]

    gs = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_grid=tuned_parameters,
        cv=cv,
        n_jobs=n_jobs,
    )
    gs = gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)

    y_pred_train = gs.predict(X_train)
    y_pred_test = gs.predict(X_test)

    train_metrics = regression_metrics(y_train, y_pred_train)
    test_metrics = regression_metrics(y_test, y_pred_test)
    print_metrics("Train", train_metrics)
    print_metrics("Test", test_metrics)

    plot_predictions(y_train, y_pred_train, y_test, y_pred_test, "Random Forest predicted PCE%", output_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest regressor on OPV PUFp features.")
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
