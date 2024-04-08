import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    breast_df = pd.read_csv('./data/wpbc.ord',sep=" ", header=None)
    breast_df.columns = [f'Feature {i}' if i < (len(breast_df.columns) - 1) else "label" for i in range(len(breast_df.columns))]


    # Encode labels
    le = LabelEncoder()
    le.fit(breast_df["label"])


    return uncertainty_experiment(breast_df, None, 'label', le, model, "Breast")


if __name__ == '__main__':
    run()
