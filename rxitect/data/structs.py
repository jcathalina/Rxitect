from enum import Enum


class RegressionQSARModel(str, Enum):
    XGB = "xgboost"
    RF = "random_forest"
    SVR = "svr"