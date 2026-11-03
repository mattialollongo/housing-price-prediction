"""Train the final model and save it to models/"""
import json

import joblib

from src.data import PROJECT_ROOT, load_housing_data, split_data, split_features_labels
from src.tune import BEST_MODEL_PATH, build_pipeline

MODEL_PATH = PROJECT_ROOT / "models" / "housing_model.pkl"


def build_model():
    """Return the winning pipeline from results/best_model.json."""
    best = json.loads(BEST_MODEL_PATH.read_text())
    print(f"Building {best['experiment']} with {best['params']}")
    return build_pipeline(best["experiment"], best["params"])


def main():
    """Train the model on the training set and save it."""
    train_set, _ = split_data(load_housing_data())
    housing, housing_labels = split_features_labels(train_set)

    model = build_model()
    model.fit(housing, housing_labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
