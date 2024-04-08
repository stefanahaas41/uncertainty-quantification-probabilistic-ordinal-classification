import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    balance_df = pd.read_csv('./data/balance+scale/balance-scale.data', names=["Class Name", "Left-Weight",
                                                                                "Left-Distance", "Right-Weight",
                                                                                "Right-Distance"])

    balance_df["Class Name"] = balance_df['Class Name'].replace('L', 1)
    balance_df['Class Name'] = balance_df['Class Name'].replace('B', 2)
    balance_df['Class Name'] = balance_df['Class Name'].replace('R', 3)


    # Encode labels
    le = LabelEncoder()
    le.fit(balance_df["Class Name"])

    enc = None

    return uncertainty_experiment(balance_df, enc, "Class Name", le, model, "Balance Scale")


if __name__ == '__main__':
    run()
