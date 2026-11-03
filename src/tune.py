"""Tune the book's random forest and the HGB pipelines under the same conditions.

Usage:
    python -m src.tune                     # all experiments
    python -m src.tune --only hgb_c rf_book

The test set is never used here.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT, load_housing_data, split_data, split_features_labels
from src.preprocessing import build_hgb_preprocessing, build_preprocessing

N_ITER = 40
SCORING = "neg_root_mean_squared_error"
SEARCH_CV = KFold(n_splits=3, shuffle=True, random_state=42)
FINAL_CV = KFold(n_splits=10, shuffle=True, random_state=42)

RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_PATH = RESULTS_DIR / "experiments.csv"
BEST_MODEL_PATH = RESULTS_DIR / "best_model.json"

HGB_PARAMS = {
    "hgb__learning_rate": loguniform(0.02, 0.3),
    "hgb__max_leaf_nodes": randint(20, 150),
    "hgb__min_samples_leaf": randint(5, 80),
    "hgb__l2_regularization": loguniform(1e-3, 10),
}

# name -> (preprocessing builder, model step name, param distributions)
EXPERIMENTS = {
    "rf_book": (build_preprocessing, "random_forest", {
        "preprocessing__geo__n_clusters": randint(3, 50),
        "random_forest__max_features": randint(2, 20),
    }),
    "hgb_a": (lambda: build_hgb_preprocessing(add_ratios=False), "hgb", HGB_PARAMS),
    "hgb_b": (lambda: build_hgb_preprocessing(add_ratios=True), "hgb", HGB_PARAMS),
    "hgb_c": (lambda: build_hgb_preprocessing(add_ratios=True, add_clusters=True), "hgb", {
        **HGB_PARAMS,
        "preprocessing__geo__n_clusters": randint(5, 50),
    }),
}


def build_pipeline(name, params=None):
    """Return the untrained pipeline of an experiment, optionally with params set."""
    build_prep, step, _ = EXPERIMENTS[name]
    if step == "random_forest":
        model = RandomForestRegressor(random_state=42)
    else:
        model = HistGradientBoostingRegressor(categorical_features=[0], max_iter=1000,
                                              early_stopping=True, random_state=42)
    pipeline = Pipeline([("preprocessing", build_prep()), (step, model)])
    if params:
        pipeline.set_params(**params)
    return pipeline


def to_builtin(params):
    """Convert numpy scalars to Python types so they can be saved as JSON."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in params.items()}


def run_search(name, X, y):
    """Run the randomized search of one experiment and return its results row."""
    _, _, param_distribs = EXPERIMENTS[name]
    search = RandomizedSearchCV(build_pipeline(name), param_distribs, n_iter=N_ITER,
                                cv=SEARCH_CV, scoring=SCORING, random_state=42, n_jobs=-1)
    start = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - start
    n_failed = int(np.isnan(search.cv_results_["mean_test_score"]).sum())
    print(f"[{name}] search RMSE (cv=3): {-search.best_score_:,.0f} "
          f"in {elapsed:.0f}s ({n_failed} failed candidates)")
    return {
        "experiment": name,
        "search_rmse_cv3": -search.best_score_,
        "cv10_rmse_mean": np.nan,
        "cv10_rmse_std": np.nan,
        "best_params": json.dumps(to_builtin(search.best_params_)),
        "search_time_s": elapsed,
        "n_iter": N_ITER,
    }


def run_cv10(name, params, X, y):
    """Re-evaluate the best pipeline of an experiment with 10-fold CV."""
    start = time.perf_counter()
    rmses = -cross_val_score(build_pipeline(name, params), X, y,
                             cv=FINAL_CV, scoring=SCORING, n_jobs=-1)
    print(f"[{name}] RMSE (cv=10): {rmses.mean():,.0f} ± {rmses.std():,.0f} "
          f"in {time.perf_counter() - start:.0f}s")
    return rmses.mean(), rmses.std()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS),
                        help="experiments to run (default: all)")
    args = parser.parse_args()
    to_run = args.only or list(EXPERIMENTS)

    train_set, _ = split_data(load_housing_data())
    X, y = split_features_labels(train_set)

    # keep the rows of the experiments that are not re-run
    if EXPERIMENTS_PATH.is_file():
        results = pd.read_csv(EXPERIMENTS_PATH)
        results = results[~results["experiment"].isin(to_run)]
    else:
        results = pd.DataFrame()

    for i, name in enumerate(to_run, start=1):
        print(f"({i}/{len(to_run)}) Searching {name} with n_iter={N_ITER}...")
        row = pd.DataFrame([run_search(name, X, y)])
        results = row if results.empty else pd.concat([results, row], ignore_index=True)

    # the final choice is made with 10-fold CV on the two best experiments
    results = results.sort_values("search_rmse_cv3").reset_index(drop=True)
    top2 = results["experiment"].head(2).tolist()
    print(f"Re-evaluating the two best experiments with cv=10: {', '.join(top2)}")
    for idx in results.index:
        name = results.at[idx, "experiment"]
        if name not in top2:
            results.loc[idx, ["cv10_rmse_mean", "cv10_rmse_std"]] = np.nan
        elif name in to_run or pd.isna(results.at[idx, "cv10_rmse_mean"]):
            params = json.loads(results.at[idx, "best_params"])
            results.loc[idx, ["cv10_rmse_mean", "cv10_rmse_std"]] = run_cv10(name, params, X, y)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(EXPERIMENTS_PATH, index=False)
    print(f"Results saved to {EXPERIMENTS_PATH.relative_to(PROJECT_ROOT)}")

    winner = results.loc[results["cv10_rmse_mean"].idxmin()]
    best_model = {
        "experiment": winner["experiment"],
        "params": json.loads(winner["best_params"]),
        "cv10_rmse_mean": winner["cv10_rmse_mean"],
        "cv10_rmse_std": winner["cv10_rmse_std"],
    }
    BEST_MODEL_PATH.write_text(json.dumps(best_model, indent=2) + "\n")
    print(f"Winner: {winner['experiment']} "
          f"(cv=10 RMSE {winner['cv10_rmse_mean']:,.0f}), "
          f"saved to {BEST_MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
