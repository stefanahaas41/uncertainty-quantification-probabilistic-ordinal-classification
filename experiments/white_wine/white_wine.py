import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment

def run(model="LGBM"):
    white_wine_df = pd.read_csv('./data/wine+quality/winequality-white.csv', sep=';')

    # Encode labels
    le = LabelEncoder()
    le.fit(white_wine_df['quality'])

    enc = None

    return uncertainty_experiment(white_wine_df, enc, 'quality', le, model, "White Wine")


if __name__ == '__main__':
    run()
