import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    tae_df = pd.read_csv('./data/teaching+assistant+evaluation/tae.data', sep=',',
                         names=['english', 'instr', 'course', 'term', 'class_size', 'label'])

    # Encode labels
    le = LabelEncoder()
    le.fit(tae_df['label'])

    return uncertainty_experiment(tae_df, None, 'label', le, model, "TAE")


if __name__ == '__main__':
    run()
