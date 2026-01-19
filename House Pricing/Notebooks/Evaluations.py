import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

from catboost import Pool, cv as cat_cv


def evaluate(X_train, X_test, y_train, y_test, models, cv=5):

    if not isinstance(models, dict):
        raise TypeError("models must be a dictionary")

    results = []

    for name, model in models.items():

        # ---------------- Fit ----------------
        model.fit(X_train, y_train)

        # ---------------- Test Evaluation ----------------
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # ---------------- Cross Validation ----------------
        if "cat" in name.lower():

            train_pool = Pool(X_train, y_train)

            params = model.get_params()
            params.update({
                "loss_function": "RMSE",
                "eval_metric": "R2",
                "verbose": False
            })

            cv_results = cat_cv(
                pool=train_pool,
                params=params,
                fold_count=cv,
                shuffle=True,
                partition_random_seed=42
            )

            # 🔑 SAFE extraction of R2 columns (version independent)
            mean_col = next(c for c in cv_results.columns if "test" in c.lower() and "r2" in c.lower() and "mean" in c.lower())
            std_col  = next(c for c in cv_results.columns if "test" in c.lower() and "r2" in c.lower() and "std" in c.lower())

            cv_mean = round(cv_results[mean_col].iloc[-1], 4)
            cv_std = round(cv_results[std_col].iloc[-1], 4)

        else:
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                scoring="r2",
                cv=cv,
                n_jobs=-1
            )

            cv_mean = round(cv_scores.mean(), 4)
            cv_std = round(cv_scores.std(), 4)

        # ---------------- Store Results ----------------
        results.append({
            "model": name,
            "test_r2": round(test_r2, 4),
            "mse": round(mse, 2),
            "cv_mean_r2": cv_mean,
            "cv_std_r2": cv_std
        })

    return (
        pd.DataFrame(results)
        .sort_values(by="test_r2", ascending=False)
        .reset_index(drop=True)
    )


def print_model_rmse(models, X_test, y_test, cat_model=None, X_test_t=None):
    """
    Prints RMSE for all trained regression models.

    Parameters
    ----------
    models : dict
        Dictionary of sklearn models or pipelines
        Example: {'lgbm': lgbm_model, 'rf': rf_model}

    X_test, y_test :
        Test data

    cat_model : CatBoostRegressor, optional
        CatBoost model trained outside Pipeline

    X_test_t : transformed X_test for CatBoost
    """

    results = []

    # sklearn-compatible models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({
            "model": name,
            "rmse": round(rmse, 2)
        })

    # CatBoost (optional)
    if cat_model is not None and X_test_t is not None:
        y_pred = cat_model.predict(X_test_t)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({
            "model": "catboost",
            "rmse": round(rmse, 2)
        })

    df = pd.DataFrame(results).sort_values("rmse")
    print(df.to_string(index=False))