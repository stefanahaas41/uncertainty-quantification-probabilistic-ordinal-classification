import pandas as pd
from category_encoders import OneHotEncoder
from scipy.io.arff import loadarff
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from util.uncertainty_experiment import uncertainty_experiment


def run(model="LGBM"):
    cmc_arff = loadarff("./data/dataset_23_cmc.arff")

    cmc_df = pd.DataFrame(cmc_arff[0])

    cmc_df['Wifes_education'] = cmc_df['Wifes_education'].astype(int)
    cmc_df['Husbands_education'] = cmc_df['Husbands_education'].astype(int)
    cmc_df['Wifes_religion'] = cmc_df['Wifes_religion'].astype(int)
    cmc_df['Wifes_now_working%3F'] = cmc_df['Wifes_now_working%3F'].astype(int)
    cmc_df['Husbands_occupation'] = cmc_df['Husbands_occupation'].astype(int)
    cmc_df['Standard-of-living_index'] = cmc_df['Standard-of-living_index'].astype(int)
    cmc_df['Media_exposure'] = cmc_df['Media_exposure'].astype(int)
    cmc_df['Contraceptive_method_used'] = cmc_df['Contraceptive_method_used'].astype(int)

    # Encode labels
    le = LabelEncoder()
    le.fit(cmc_df['Contraceptive_method_used'])

    # Encode categorical data
    enc = None

    return uncertainty_experiment(cmc_df, enc, 'Contraceptive_method_used', le, model, "cmc")


if __name__ == '__main__':
    run()
