import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    abalone_df = pd.read_csv('./data/abalone.ord',sep=" ", header=None)
    abalone_df.columns = [f'Feature {i}' if i < (len(abalone_df.columns) - 1) else "label" for i in range(len(abalone_df.columns))]


    # Encode labels
    le = LabelEncoder()
    le.fit(abalone_df["label"])


    return uncertainty_experiment(abalone_df, None, 'label', le, model, "Abalone")


if __name__ == '__main__':
    run()
