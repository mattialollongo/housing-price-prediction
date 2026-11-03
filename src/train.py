"""Train the final model and save it to models/"""
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT, load_housing_data, split_data, split_features_labels
from src.preprocessing import build_preprocessing

MODEL_PATH = PROJECT_ROOT / "models" / "housing_model.pkl"

# best hyperparameters from the randomized search in the notebook
BEST_PARAMS = {
    "preprocessing__geo__n_clusters": 45,
    "random_forest__max_features": 9,
}


def build_model():
    """Return the full pipeline with the best hyperparameters."""
    model = Pipeline([
        ("preprocessing", build_preprocessing()),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ])
    model.set_params(**BEST_PARAMS)
    return model


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
