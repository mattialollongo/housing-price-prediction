"""Download and load data, creates split"""
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


housing_full = load_housing_data()

housing_full["income_cat"] = pd.cut(
    housing_full["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
    )

strat_train_test, strat_test_set = train_test_split(
    housing_full,
    test_size=0.2, 
    stratify=housing_full["income_cat"], 
    random_state=42
    )

