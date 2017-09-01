from wiki_forecasting import validation_utils

from sklearn.model_selection import GroupKFold
import xgboost as xgb
import pandas as pd

if __name__ == "__main__":
    model = xgb.XGBRegressor()
    cv = GroupKFold(2)
    cv_scores, test_scores, imps = validation_utils.validate_model(model, cv=cv, longest_gap=60)
    pd.Series(cv_scores).to_csv("cv_scores.csv")
    pd.Series(test_scores).to_csv("cv_scores.csv")
    pd.DataFrame(imps).to_csv("imps.csv")
