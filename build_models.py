from wiki_forecasting import validation_utils

import xgboost as xgb

if __name__ == "__main__":
    model = xgb.XGBRegressor()
    cv_scores, test_scores, imps = validation_utils.fit_model(model, longest_gap=60)
