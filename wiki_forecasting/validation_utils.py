from wiki_forecasting.data_utils import prepare_data, prepare_future_data, project_root

import os
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
import pickle as pkl
import pandas as pd
import pickle as pkl


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)


def validate_model(model, cv, longest_gap):
    cv_scores = []
    test_scores = []
    imps = []
    for prediction_gap in range(1, longest_gap+1):
        X, y, sums, X_test, y_test, sums_test, df = prepare_data(prediction_gap)
        X["date_code"] = X["date_code"] // 20
        preds = cross_val_predict(model,
                                  X.drop(["date", "date_code"], axis=1), np.log(y + 1),
                                  groups=X["date_code"], cv=cv)
        cv_score = smape(y, np.exp(preds) - 1)
        cv_scores.append(cv_score)

        model = model.fit(
            X.drop(["date", "date_code"], axis=1),
            np.log(y + 1)
        )
        preds = model.predict(X_test.drop(["date", "date_code"], axis=1))
        b = model.booster()
        imps.append(b.get_fscore())

        test_score = smape(y_test, np.exp(preds) - 1)
        test_scores.append(test_score)
        print(prediction_gap, cv_score, test_score)
    return cv_scores, test_scores, imps


def fit_model(model, longest_gap):
    for prediction_gap in range(1, longest_gap+1):
        print(prediction_gap)
        X, y, sums, X_test, y_test, sums_test, df = prepare_data(prediction_gap)
        X = pd.concat([X, X_test])
        y = pd.concat([y, y_test])
        model = model.fit(
            X.drop(["date", "date_code"], axis=1),
            np.log(y + 1)
        )
        with open(os.path.join(project_root, "models", "xgb.{}.pkl".format(prediction_gap)), "wb") as f:
            pkl.dump(model, f)


def predict_future(longest_gap):
    for i, X in enumerate(prepare_future_data(longest_gap)):
        print("Predicting gap {}".format(i))
        with open(os.path.join(project_root, "models", "xgb.{}.pkl".format(i)), "rb") as f:
            model = pkl.load(f)
        p = pd.DataFrame({"preds": np.exp(model.predict(X.drop(["date", "date_code"], axis=1))) - 1,
                          "date": X["date"]})
        yield p
