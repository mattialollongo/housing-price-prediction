# Housing price prediction

## Overview

In this project I predict the median house value of California districts from
census data. It is a regression task, and models are evaluated with the RMSE.
The project is based on Chapter 2 of *Hands-On Machine Learning* by Aurélien Géron,
and covers all the main steps of an end-to-end ML project: data exploration,
a preprocessing pipeline, model selection, fine-tuning and evaluation on the test set.

The book ends up with a random forest. To give it a twist, I also tested a
histogram-based gradient boosting model (HGB) to see if it could do better.
After tuning both models, the random forest still came out on top
(CV RMSE 42,118 vs 44,352), with a final test RMSE of about 41,423 USD.

## Dataset

California Housing dataset from the 1990 California census: 20,640 districts, 10 columns.

- 9 features: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity` (categorical)
- Target: `median_house_value`
- `total_bedrooms` has 207 missing values

The data is downloaded automatically into `datasets/` on the first run.

## Approach

<!-- Draft: to be rewritten -->
- Stratified train/test split (80/20) on income categories
- Exploration: geographic plots, correlations, combined ratio features
- Preprocessing pipeline:
  - median imputation, standard scaling
  - ratio features (bedrooms/rooms, rooms/household, people/household)
  - log transform of skewed features
  - geographic cluster similarity (k-means + RBF kernel)
  - one-hot encoding of `ocean_proximity`
- Model selection with 10-fold cross-validation: linear regression, decision tree, random forest, hist gradient boosting
- Fine-tuning of random forest and HGB with randomized search
- Final evaluation on the test set, with a bootstrap 95% confidence interval

## Results

Cross-validation RMSE (10 folds, default hyperparameters):

| Model | Mean RMSE | Std |
|---|---:|---:|
| Linear regression | 70,003 | 4,182 |
| Decision tree | 67,013 | 1,460 |
| Random forest | 47,125 | 1,069 |
| Hist gradient boosting | 45,379 | 1,100 |

After fine-tuning with randomized search (3 folds), the random forest came out ahead: 42,118 against 44,353 for the HGB.

Final model on the test set:

| Model | Test RMSE | 95% CI |
|---|---:|---:|
| Random forest (`n_clusters=45`, `max_features=9`) | 41,423 | 39,489 – 43,666 |

## Project structure

```
housing-price-prediction/
├── notebooks/
│   └── 01_exploration.ipynb   # exploration, model selection, fine-tuning
├── src/
│   ├── data.py                # download, train/test split
│   ├── preprocessing.py       # preprocessing pipeline
│   ├── train.py               # trains and saves the final model
│   └── evaluate.py            # evaluates the model on the test set
├── requirements.txt
└── README.md
```

## How to run

From the project root, run the commands in this order:

```bash
pip install -r requirements.txt
python -m src.tune       # downloads the data, runs all the searches, saves results/experiments.csv and results/best_model.json
python -m src.train      # trains the winner on the whole training set, saves models/housing_model.pkl
python -m src.evaluate   # evaluates the model on the test set once, saves results/test_results.json
```

`python -m src.tune --only hgb_c` re-runs a single experiment (`rf_book`, `hgb_a`, `hgb_b`, `hgb_c`)
and keeps the results of the others.

## Credits

- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch*, chapter 2
- California Housing dataset: R. Kelley Pace and Ronald Barry, "Sparse Spatial Autoregressions", *Statistics & Probability Letters*, 33 (1997) 291–297
