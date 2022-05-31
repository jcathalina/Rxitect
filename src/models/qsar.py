import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    reg = xgb.XGBRegressor(
        tree_method="hist",
        eval_metric=mean_absolute_error,
    )
    reg.fit(X, y, eval_set=[(X, y)])
