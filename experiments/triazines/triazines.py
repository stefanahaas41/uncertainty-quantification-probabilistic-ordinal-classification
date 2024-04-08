import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    triazines_df = pd.read_csv('./data/triazines.ord',sep=" ", header=None)
    triazines_df.columns = [f'Feature {i}' if i < (len(triazines_df.columns) - 1) else "label" for i in range(len(triazines_df.columns))]


    # Encode labels
    le = LabelEncoder()
    le.fit(triazines_df["label"])


    return uncertainty_experiment(triazines_df, None, 'label', le, model, "Triazines")


if __name__ == '__main__':
    run()
