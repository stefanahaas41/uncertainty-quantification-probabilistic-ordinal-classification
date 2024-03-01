import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    esl_df = loadarff('./data/ESL.arff')
    esl_df = pd.DataFrame(esl_df[0])
    # Encode labels
    le = LabelEncoder()
    le.fit(esl_df['out1'])

    enc = None

    return uncertainty_experiment(esl_df, enc, 'out1', le, model,"ESL")

if __name__ == '__main__':
    run()
