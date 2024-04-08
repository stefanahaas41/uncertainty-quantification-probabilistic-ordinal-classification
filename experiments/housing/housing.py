import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    housing_df = pd.read_csv('./data/housing.ord',sep=" ", header=None)
    housing_df.columns = [f'Feature {i}' if i < (len(housing_df.columns) - 1) else "label" for i in range(len(housing_df.columns))]


    # Encode labels
    le = LabelEncoder()
    le.fit(housing_df["label"])


    return uncertainty_experiment(housing_df, None, 'label', le, model, "Housing")


if __name__ == '__main__':
    run()
