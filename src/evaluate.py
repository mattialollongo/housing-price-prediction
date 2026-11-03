"""Evaluate the saved model on the test set"""
import json

import joblib
import numpy as np
from scipy import stats
from sklearn.metrics import root_mean_squared_error

from src.data import PROJECT_ROOT, load_housing_data, split_data, split_features_labels
from src.train import MODEL_PATH
from src.tune import RESULTS_DIR

TEST_RESULTS_PATH = RESULTS_DIR / "test_results.json"


def rmse(squared_errors):
    """Return the RMSE given the squared errors."""
    return np.sqrt(np.mean(squared_errors))


def main():
    """Compute the test RMSE and its 95% confidence interval, and save them."""
    model = joblib.load(MODEL_PATH)
    _, test_set = split_data(load_housing_data())
    X_test, y_test = split_features_labels(test_set)

    final_predictions = model.predict(X_test)
    final_rmse = root_mean_squared_error(y_test, final_predictions)

    squared_errors = (final_predictions - y_test) ** 2
    boot_result = stats.bootstrap([squared_errors], rmse,
                                  confidence_level=0.95, random_state=42)
    rmse_lower, rmse_upper = boot_result.confidence_interval
    print(f"Test RMSE: {final_rmse:,.0f} "
          f"(95% CI: {rmse_lower:,.0f} – {rmse_upper:,.0f})")

    # the last step of a pipeline is the estimator
    estimator = model[-1] if hasattr(model, "steps") else model
    test_results = {
        "model": type(estimator).__name__,
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "n_test_samples": len(y_test),
        "test_rmse": final_rmse,
        "ci_95_lower": rmse_lower,
        "ci_95_upper": rmse_upper,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_RESULTS_PATH.write_text(json.dumps(test_results, indent=2) + "\n")
    print(f"Results saved to {TEST_RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
