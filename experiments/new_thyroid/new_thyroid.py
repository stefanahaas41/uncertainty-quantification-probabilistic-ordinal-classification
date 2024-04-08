import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    new_thyroid_df = pd.read_csv('./data/thyroid+disease/new-thyroid.data', sep=',',
                              names=["Class","T3-resin","Total Serum thyroxin","Total serum triiodothyronine",
                                      "basal thyroid-stimulating hormone (TSH)","Maximal absolute difference of TSH"])
    # Encode labels
    le = LabelEncoder()
    le.fit(new_thyroid_df['Class'])

    enc = None

    return uncertainty_experiment(new_thyroid_df, enc, 'Class', le, model, "New Thyroid")


if __name__ == '__main__':
    run()
