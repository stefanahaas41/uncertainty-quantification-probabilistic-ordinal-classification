import csv

import pandas as pd
import seaborn as sns

from experiments.automobile import automobile
from experiments.balance import balance
from experiments.era import era
from experiments.esl import esl
from experiments.eucalyptus import eucalyptus
from experiments.heart import heart
from experiments.lev import lev
from experiments.new_thyroid import new_thyroid
from experiments.red_wine import red_wine
from experiments.swd import swd
from experiments.tae import tae
from experiments.white_wine import white_wine

MODEL = "LGBM"


def main():
    sns.set_theme(context='paper', style='ticks', font_scale=1.4, font="serif", rc={
        "text.usetex": True
    })

    # Run experiments
    new_thyroid_prr_df, new_thyroid_prr_raw_df, new_thyroid_raw_prr_data_df = new_thyroid.run(MODEL)

    balance_scale_prr_df, balance_scale_prr_raw_df, balance_raw_prr_data_df = balance.run(MODEL)

    auto_prr_df, auto_prr_raw_df, automobile_raw_prr_data_df = automobile.run(MODEL)

    swd_prr_df, swd_prr_raw_df, swd_raw_prr_data_df = swd.run(MODEL)

    era_prr_df, era_prr_raw_df, era_raw_prr_data = era.run(MODEL)

    esl_prr_df, esl_prr_raw_df, esl_raw_prr_data = esl.run(MODEL)

    lev_prr_df, lev_prr_raw_df, lev_raw_prr_data = lev.run(MODEL)

    eucalyptus_prr_df, eucalyptus_prr_raw_df, eucalyptus_raw_prr_data = eucalyptus.run(MODEL)

    tae_prr_df, tae_prr_raw_df, tae_prr_data = tae.run(MODEL)

    red_wine_prr_df, red_wine_prr_raw_df, red_wine_raw_prr_data = red_wine.run(MODEL)

    white_wine_prr_df, white_wine_prr_raw_df, white_wine_raw_prr_data = white_wine.run(MODEL)

    heart_prr_df, heart_prr_raw_df, heart_raw_prr_data = heart.run(MODEL)

    # Build final results
    result_df = pd.concat([
        new_thyroid_prr_df,
        balance_scale_prr_df,
        auto_prr_df,
        eucalyptus_prr_df,
        tae_prr_df,
        heart_prr_df,
        swd_prr_df,
        era_prr_df,
        esl_prr_df,
        lev_prr_df,
        red_wine_prr_df,
        white_wine_prr_df
    ])

    result_raw_df = pd.concat([
        new_thyroid_prr_raw_df,
        balance_scale_prr_raw_df,
        auto_prr_raw_df,
        eucalyptus_prr_raw_df,
        tae_prr_raw_df,
        heart_prr_raw_df,
        swd_prr_raw_df,
        era_prr_raw_df,
        esl_prr_raw_df,
        lev_prr_raw_df,
        red_wine_prr_raw_df,
        white_wine_prr_raw_df
    ])

    result_prr_raw_df = pd.concat([
        new_thyroid_raw_prr_data_df,
        balance_raw_prr_data_df,
        automobile_raw_prr_data_df,
        eucalyptus_raw_prr_data,
        tae_prr_raw_df,
        heart_prr_raw_df,
        swd_raw_prr_data_df,
        era_raw_prr_data,
        esl_raw_prr_data,
        lev_raw_prr_data,
        red_wine_raw_prr_data,
        white_wine_raw_prr_data
    ])

    result_df.to_csv(MODEL + "_prr_result.csv", sep='&', lineterminator="\\\\\n", quoting=csv.QUOTE_NONE, quotechar='',
                     escapechar=' ', index=False)

    result_raw_df.to_csv(MODEL + "_prr_result_raw.csv", index=False)

    result_prr_raw_df.to_csv(MODEL + "_prr_result_total_raw.csv", index=False)


if __name__ == '__main__':
    main()
