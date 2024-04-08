import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    pyrim_df = pd.read_csv('./data/pyrim.ord',sep=" ", header=None)
    pyrim_df.columns = [f'Feature {i}' if i < (len(pyrim_df.columns) - 1) else "label" for i in range(len(pyrim_df.columns))]


    # Encode labels
    le = LabelEncoder()
    le.fit(pyrim_df["label"])


    return uncertainty_experiment(pyrim_df, None, 'label', le, model, "Pyrimidines")


if __name__ == '__main__':
    run()
