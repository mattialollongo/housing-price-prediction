"""Download and load data, creates split"""
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_housing_data():
    """Download the housing dataset if needed and return it as a DataFrame."""
    tarball_path = DATASETS_DIR / "housing.tgz"
    if not tarball_path.is_file():
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=DATASETS_DIR, filter="data")
    return pd.read_csv(DATASETS_DIR / "housing" / "housing.csv")


def split_data(housing):
    """Split the data into train and test sets, stratified on income."""
    housing = housing.copy()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
        )

    strat_train_set, strat_test_set = train_test_split(
        housing,
        test_size=0.2,
        stratify=housing["income_cat"],
        random_state=42
        )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set


def split_features_labels(df):
    """Separate the features from the median_house_value labels."""
    features = df.drop("median_house_value", axis=1)
    labels = df["median_house_value"].copy()
    return features, labels
