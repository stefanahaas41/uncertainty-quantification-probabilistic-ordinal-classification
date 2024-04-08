
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from util.measures.binary_decompositions import binary_ordinal_entropy, binary_ordinal_margin
from util.measures.coefficient_of_agreement import coefficient_of_agreement_std
from util.measures.consensus import consensus
from util.measures.dfu import dfu
from util.measures.empirical_risk import empirical_risk, risk_of_prediction
from util.measures.ordinal_consensus_l1 import ordinal_consensus
from util.measures.ordinal_variation import ordinal_variation_l2
from util.measures.variance import variance_for_probabilities
from util.rejection_curve_local import rejection_curve, plot_rejection_curve_df


def uncertainty_experiment(data_df, enc, label, le, model, dataset):
    print("Experiment: " + dataset)

    overall_result_rejection_curve = {
        "Rejection": [],
        "Performance": [],
        "Metric": [],
        "Measure": []
    }

    overall_result_prr = {
        "ACC": [],
        "MAE": [],
        "MSE": []
    }

    overall_prr_plots = {}

    X = data_df.copy()
    y = X.pop(label)

    kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"Fold {i}:")

        X_train = data_df.iloc[train_index]
        X_test = data_df.iloc[test_index]

        y_train = X_train.pop(label)
        y_test = X_test.pop(label)

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()

        y_train = le.transform(y_train)

        if enc:
            X_train = enc.fit_transform(X_train)
            X_test = enc.transform(X_test)

        u_measures = main_experiment_part(X_test, X_train, le, model, overall_result_prr,
                                          overall_result_rejection_curve, overall_prr_plots, y_test, y_train)

    result_prr_df, result_prr_raw_df, combined_raw_prr_data = prepare_prr_result(dataset, overall_result_prr,
                                                                                 u_measures)

    # Save rejection curves
    overall_result_df = pd.DataFrame(overall_result_rejection_curve)
    overall_result_df.to_csv(model + "_" + dataset + "_result.csv", index=False)

    # Save PRR plots
    for metric_key, metric_values in overall_prr_plots.items():
        for measure_key, measure_values in metric_values.items():
            df = pd.DataFrame(overall_prr_plots[metric_key][measure_key])
            if metric_key == 'ACC':
                x_label = 'MCR'
            else:
                x_label = metric_key
            plot_rejection_curve_df(df, ("ci", 95), x_label,
                                    "./prr_plots/" + dataset + "_" + x_label + "_" + measure_key + ".pdf",
                                    measure_key)

    return result_prr_df, result_prr_raw_df, combined_raw_prr_data


def prepare_prr_result(dataset, overall_result_prr, u_measures):
    # Save PRR results
    acc_mean, acc_std = metric_mean_stdev_methods(overall_result_prr, "ACC")
    mae_mean, mae_std = metric_mean_stdev_methods(overall_result_prr, "MAE")
    mse_mean, mse_std = metric_mean_stdev_methods(overall_result_prr, "MSE")

    result_prr_raw_mean = {
        "ACC": [f'{round(acc_mean[i], 4)}' for i in range(len(acc_mean))],
        "MAE": [f'{round(mae_mean[i], 4)}' for i in range(len(mae_mean))],
        "MSE": [f'{round(mse_mean[i], 4)}' for i in range(len(mse_mean))]
    }

    result_prr = {
        "ACC": [f'{round(acc_mean[i], 4)} \\textpm {round(acc_std[i], 4)}' if i != np.argmax(acc_mean)
                else f'\\textbf{{{round(acc_mean[i], 4)}}}\\textpm {round(acc_std[i], 4)}'
                for i in range(len(acc_mean))],
        "MAE": [f'{round(mae_mean[i], 4)} \\textpm {round(mae_std[i], 4)}' if i != np.argmax(mae_mean)
                else f'\\textbf{{{round(mae_mean[i], 4)}}}\\textpm {round(mae_std[i], 4)}'
                for i in range(len(mae_mean))],
        "MSE": [f'{round(mse_mean[i], 4)} \\textpm {round(mse_std[i], 4)}' if i != np.argmax(mse_mean)
                else f'\\textbf{{{round(mse_mean[i], 4)}}}\\textpm {round(mse_std[i], 4)}'
                for i in range(len(mse_mean))]
    }
    measures = list(u_measures.keys())
    measures.remove("Random")

    acc_results = overall_result_prr["ACC"]
    mae_results = overall_result_prr["MAE"]
    mse_results = overall_result_prr["MSE"]

    acc_df = pd.DataFrame(acc_results, columns=measures)
    acc_df["Metric"] = "ACC"
    acc_df["Dataset"] = dataset
    acc_df.set_index("Metric")
    mae_df = pd.DataFrame(mae_results, columns=measures)
    mae_df["Metric"] = "MAE"
    mae_df.set_index("Metric")
    mae_df["Dataset"] = dataset
    mse_df = pd.DataFrame(mse_results, columns=measures)
    mse_df["Metric"] = "MSE"
    mse_df.set_index("Metric")
    mse_df["Dataset"] = dataset

    combined_raw_prr_data = pd.concat([acc_df, mae_df, mse_df], ignore_index=True)

    # Latex format
    result_prr_df = pd.DataFrame.from_dict(result_prr, orient='index', columns=measures)
    result_prr_df.insert(loc=0, column="Dataset", value=dataset)
    result_prr_df.insert(loc=1, column="Metric", value=result_prr_df.index)
    result_prr_df.reset_index(drop=True, inplace=True)

    # Raw means
    result_prr_raw_df = pd.DataFrame.from_dict(result_prr_raw_mean, orient='index', columns=measures)
    result_prr_raw_df.insert(loc=0, column="Dataset", value=dataset)
    result_prr_raw_df.insert(loc=1, column="Metric", value=result_prr_raw_df.index)
    result_prr_raw_df.reset_index(drop=True, inplace=True)

    return result_prr_df, result_prr_raw_df, combined_raw_prr_data


def main_experiment_part(X_test, X_train, le, model, overall_result_prr, overall_result_rejection_curve,
                         overall_prr_plots, y_test,
                         y_train):
                         
    if model == "LGBM":
        clf = LGBMClassifier(random_state=42)
    else:
        raise Exception("Unknown Model")

    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)

    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred = le.inverse_transform(y_pred)

    eval_result(y_pred, y_test)

    # Uncertainty quantification
    u_confidence = 1 - np.amax(y_pred_proba, axis=1)
    u_margin = 1 - (np.amax(y_pred_proba, axis=1) - (np.partition(y_pred_proba, -2, axis=1)[:, -2]))
    u_entropy = entropy(y_pred_proba, axis=1)
    u_variance = np.apply_along_axis(variance_for_probabilities, 1, y_pred_proba)
    u_consensus_leik = np.apply_along_axis(ordinal_consensus, 1, y_pred_proba)
    u_consensus_blair = 1 - np.apply_along_axis(ordinal_variation_l2, 1, y_pred_proba)
    u_consensus_tastle = 1 - np.apply_along_axis(consensus, 1, y_pred_proba)
    u_agreement = 1 - np.apply_along_axis(coefficient_of_agreement_std, 1, y_pred_proba)
    u_dfu = np.apply_along_axis(dfu, 1, y_pred_proba)
    u_ordinal_binary_entropy = np.apply_along_axis(binary_ordinal_entropy, 1, y_pred_proba)
    u_ordinal_binary_margin = np.apply_along_axis(binary_ordinal_margin, 1, y_pred_proba)
    y_risk_abs = empirical_risk(y_pred_proba, "l1")
    u_absolute = risk_of_prediction(y_risk_abs, y_pred_proba, "l1")
    y_risk_squared = empirical_risk(y_pred_proba, "l2")
    u_squared = risk_of_prediction(y_risk_squared, y_pred_proba, "l2")
    random = np.random.random_sample(y_pred.shape)

    u_measures = {
        "Confidence": u_confidence,
        "Margin": u_margin,
        "Entropy": u_entropy,
        "Variance": u_variance,
        "Consensus (Tastle)": u_consensus_tastle,
        "Consensus (Leik)": u_consensus_leik,
        "Consensus (Blair)": u_consensus_blair,
        "Binary (Entropy)": u_ordinal_binary_entropy,
        "Binary (Margin)": u_ordinal_binary_margin,
        "Risk (Absolute)": u_absolute,
        "Risk (Squared)": u_squared,
        "Agreement": u_agreement,
        "Dfu": u_dfu,
        "Random": random
    }

    metrics = ["ACC", "MAE", "MSE"]

    result_fold, prr_fold, prr_plots = rejection_curve(metrics, u_measures, y_pred_proba, y_test, le)

    overall_result_prr["ACC"].append(prr_fold["ACC"])
    overall_result_prr["MAE"].append(prr_fold["MAE"])
    overall_result_prr["MSE"].append(prr_fold["MSE"])

    overall_result_rejection_curve["Rejection"].extend(result_fold["Rejection"])
    overall_result_rejection_curve["Performance"].extend(result_fold["Performance"])
    overall_result_rejection_curve["Measure"].extend(result_fold["Measure"])
    overall_result_rejection_curve["Metric"].extend(result_fold["Metric"])

    for metric_key, metric_values in prr_plots.items():
        if metric_key not in overall_prr_plots:
            overall_prr_plots[metric_key] = {}
        for measure_key, measure_values in metric_values.items():
            if measure_key not in overall_prr_plots[metric_key]:
                overall_prr_plots[metric_key][measure_key] = {}
            for plot_key, plot_values in measure_values.items():
                if plot_key not in overall_prr_plots[metric_key][measure_key]:
                    overall_prr_plots[metric_key][measure_key][plot_key] = []
                overall_prr_plots[metric_key][measure_key][plot_key].extend(plot_values)

    return u_measures


def eval_result(y_pred, y_test):
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred,
                            weights="quadratic")
    print("ACC", acc)
    print("MAE", mae)
    print("MSE", mse)
    print("QWK", qwk)

    cf_matrix = confusion_matrix(y_test, y_pred, normalize=None)
    print(cf_matrix)


def metric_mean_stdev_methods(overall_result_prr, metric):
    prr = np.array(overall_result_prr[metric])
    prr = prr[~np.isnan(prr).any(axis=1)]  # remove rows with nans
    prr_mean = np.mean(prr, axis=0)
    prr_std = np.std(prr, axis=0)
    return prr_mean, prr_std
