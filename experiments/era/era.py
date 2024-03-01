import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    era_df = loadarff('./data/ERA.arff')
    era_df = pd.DataFrame(era_df[0])
    # Encode labels
    le = LabelEncoder()
    le.fit(era_df['out1'])

    enc = None

    return uncertainty_experiment(era_df, enc, 'out1', le, model,"ERA")


if __name__ == '__main__':
    run()
