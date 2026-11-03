# Housing price prediction

## How to run

From the project root:

```bash
pip install -r requirements.txt
python -m src.train      # downloads the data, trains the model, saves models/housing_model.pkl
python -m src.evaluate   # prints the test RMSE with a 95% confidence interval
```
