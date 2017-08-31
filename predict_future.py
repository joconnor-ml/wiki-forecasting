from wiki_forecasting import validation_utils


if __name__ == "__main__":
    for i, df in enumerate(validation_utils.predict_future(longest_gap=60)):
        df.to_csv("predictions/pred.{}.csv".format(i))