import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    red_wine_df = pd.read_csv('./data/wine+quality/winequality-red.csv', sep=';')

    # Encode labels
    le = LabelEncoder()
    le.fit(red_wine_df['quality'])

    enc = None

    return uncertainty_experiment(red_wine_df, enc, 'quality', le, model, "Red Wine")


if __name__ == '__main__':
    run()
