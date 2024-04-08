import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    swd_df = loadarff('./data/SWD.arff')
    swd_df = pd.DataFrame(swd_df[0])

    # Encode labels
    le = LabelEncoder()
    le.fit(swd_df['Out1'])

    enc = None

    return uncertainty_experiment(swd_df, enc, 'Out1', le, model, "SWD")


if __name__ == '__main__':
    run()
