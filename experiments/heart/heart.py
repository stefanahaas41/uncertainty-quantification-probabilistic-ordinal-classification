import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    heart_disease_df = pd.read_csv("./data/heart+disease/processed.cleveland.data",
                                   names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                                          "oldpeak", "slope", "ca", "thal", "num"])

    # Replace missing values
    heart_disease_df = heart_disease_df.replace('?', -10.0)
    heart_disease_df[["ca", "thal"]] = heart_disease_df[["ca", "thal"]].astype(float)

    # Encode labels
    le = LabelEncoder()
    le.fit(heart_disease_df['num'])

    enc = None

    return uncertainty_experiment(heart_disease_df, enc, 'num', le, model, "Heart (CLE)")


if __name__ == '__main__':
    run()
