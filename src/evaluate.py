"""Evaluate the saved model on the test set"""
import joblib
import numpy as np
from scipy import stats
from sklearn.metrics import root_mean_squared_error

from src.data import load_housing_data, split_data, split_features_labels
from src.train import MODEL_PATH


def rmse(squared_errors):
    """Return the RMSE given the squared errors."""
    return np.sqrt(np.mean(squared_errors))


def main():
    """Compute the test RMSE and its 95% confidence interval."""
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


if __name__ == "__main__":
    main()
