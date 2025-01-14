import os
import random
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
from dlordinal.losses import WKLoss, BetaCrossEntropyLoss, TriangularCrossEntropyLoss
from lightgbm import LGBMClassifier
from pycalib.metrics import ECE, brier_score
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, \
    cohen_kappa_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from torch import nn
from torch.nn import Softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util.estimators.BinaryOrdinalClassifier import BinaryOrdinalClassifier
from util.measures.binary_decompositions import binary_ordinal_entropy, binary_ordinal_margin, binary_ordinal_variance
from util.measures.coefficient_of_agreement import coefficient_of_agreement_std
from util.measures.consensus import consensus
from util.measures.dfu import dfu
from util.measures.empirical_risk import empirical_risk, risk_of_prediction
from util.measures.ordinal_consensus_l1 import ordinal_consensus
from util.measures.ordinal_variation import ordinal_variation_l2
from util.measures.variance import variance_for_probabilities
from util.rejection_curve_local import rejection_curve

performance = {}

torch.set_default_dtype(torch.float32)


# Function to set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seeds for reproducibility
set_seed(42)


class MLPModule(nn.Module):
    def __init__(
            self,
            input_units=20,
            output_units=2,
            hidden_units=[128, 64],
            apply_softmax=False,
            weight_initialization=True
    ):
        super().__init__()

        # Define the layers
        in_layer = nn.Linear(in_features=input_units, out_features=hidden_units[0], bias=True)
        hidden_layer1 = nn.Linear(in_features=hidden_units[0], out_features=hidden_units[1], bias=True)
        out_layer2 = nn.Linear(in_features=hidden_units[1], out_features=output_units, bias=True)

        # Initialize weights
        if weight_initialization:
            nn.init.xavier_uniform_(in_layer.weight)
            nn.init.constant_(in_layer.bias, 0)

            nn.init.xavier_uniform_(hidden_layer1.weight)
            nn.init.constant_(hidden_layer1.bias, 0)

            nn.init.xavier_uniform_(out_layer2.weight)
            nn.init.constant_(out_layer2.bias, 0)

        # Define the sequence of layers
        layers = [
            in_layer,
            nn.ReLU(),
            hidden_layer1,
            nn.ReLU(),
            out_layer2
        ]

        # Optionally apply Softmax
        if apply_softmax:
            layers.append(Softmax(dim=-1))

        self.sequential = nn.Sequential(
            *layers
        )

    def forward(self, X):
        X = self.sequential(X)
        return X


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

    performance[dataset] = {"ACC": [],
                            "MAE": [],
                            "MSE": [],
                            "QWK": [],
                            "NLL": [],
                            "BRIER": [],
                            "ECE": []
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
            pipe = Pipeline(steps=[
                ('encoder', enc),
                ('scaler', StandardScaler())
            ])
            X_train = pipe.fit_transform(X_train)
            X_test = pipe.transform(X_test)
        else:
            pipe = Pipeline(steps=[('scaler', StandardScaler())])
            X_train = pipe.fit_transform(X_train)
            X_test = pipe.transform(X_test)

        u_measures = main_experiment_part(X_test, X_train, le, model, overall_result_prr,
                                          overall_result_rejection_curve, overall_prr_plots, y_test, y_train, dataset,
                                          performance)

    save_performance_data(dataset, model)

    result_prr_df, result_prr_raw_df, combined_raw_prr_data = prepare_prr_result(dataset, overall_result_prr,
                                                                                 u_measures)

    # Save rejection curves
    overall_result_df = pd.DataFrame(overall_result_rejection_curve)
    overall_result_df.to_csv('./results/' + model + "/" + dataset + "_result.csv", index=False)

    return result_prr_df, result_prr_raw_df, combined_raw_prr_data


def save_performance_data(dataset, model):
    mean_acc = mean(performance[dataset]["ACC"])
    std_acc = stdev(performance[dataset]["ACC"])
    mean_mae = mean(performance[dataset]["MAE"])
    std_mae = stdev(performance[dataset]["MAE"])
    mean_mse = mean(performance[dataset]["MSE"])
    std_mse = stdev(performance[dataset]["MSE"])
    mean_qwk = mean(performance[dataset]["QWK"])
    std_qwk = stdev(performance[dataset]["QWK"])
    mean_nll = mean(performance[dataset]["NLL"])
    std_nll = stdev(performance[dataset]["NLL"])
    mean_brier = mean(performance[dataset]["BRIER"])
    std_brier = stdev(performance[dataset]["BRIER"])
    mean_ece = mean(performance[dataset]["ECE"])
    std_ece = stdev(performance[dataset]["ECE"])

    file_path = './results/' + model + "/performance.txt"

    # Check if the file exists and has content before writing the header
    header = not os.path.isfile(file_path) or os.path.getsize(file_path) == 0
    df = pd.DataFrame([[dataset, mean_acc, std_acc, mean_mae, std_mae, mean_mse, std_mse, mean_qwk, std_qwk, mean_nll,
                        std_nll, mean_brier, std_brier, mean_ece, std_ece]],
                      columns=['DATA', 'ACC', 'STD ACC', 'MAE', 'STD MAE', 'MSE', 'STD MSE', 'QWK', 'STD QWK',
                               'NLL', 'NLL STD', 'BRIER', 'BRIER STD', 'ECE', 'ECE STD'])
    df.to_csv(file_path, mode='a', header=header, index=False)


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
    measures.remove("random")

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
                         y_train, dataset, performance):
    if model == "LGBM":
        clf = LGBMClassifier(random_state=42)
    elif model == "SIMPLE":
        clf = BinaryOrdinalClassifier(LGBMClassifier, le.transform(le.classes_[:-1]), {"random_state": 42})
    elif model == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=(128, 64),
                            random_state=42)
    elif model == "SIMPLE_MLP":
        clf = BinaryOrdinalClassifier(MLPClassifier, le.transform(le.classes_[:-1]), {"hidden_layer_sizes": (128, 64),
                                                                                      "random_state": 42})
    elif model == 'QWK':

        X_test, X_train, num_classes, num_features = prepare_mlp(X_test, X_train, y_train)

        clf = NeuralNetClassifier(
            MLPModule,
            module__input_units=num_features,
            module__output_units=num_classes,
            module__apply_softmax=True,
            module__weight_initialization=False,
            max_epochs=200,
            batch_size=min(200, len(X_train)),
            optimizer=Adam,
            optimizer__weight_decay=0.0001,
            lr=0.001,
            train_split=ValidSplit(0.1),  # Use 10% of the data for validation
            criterion=WKLoss(num_classes=num_classes),
            callbacks=[
                EarlyStopping(patience=10, monitor="valid_loss"),
                LRScheduler(policy=ReduceLROnPlateau, patience=5, factor=0.5)
            ]
        )

    elif model == 'BETA':

        X_test, X_train, num_classes, num_features = prepare_mlp(X_test, X_train, y_train)

        clf = NeuralNetClassifier(
            MLPModule,
            module__input_units=num_features,
            module__output_units=num_classes,
            module__apply_softmax=True,
            module__weight_initialization=False,
            max_epochs=200,
            batch_size=min(200, len(X_train)),
            optimizer=Adam,
            optimizer__weight_decay=0.0001,
            lr=0.001,
            train_split=ValidSplit(0.1),  # Use 10% of the data for validation
            criterion=BetaCrossEntropyLoss(num_classes=num_classes),
            callbacks=[
                EarlyStopping(patience=10, monitor="valid_loss"),
                LRScheduler(policy=ReduceLROnPlateau, patience=5, factor=0.5)
            ]
        )
    elif model == 'TRI':

        X_test, X_train, num_classes, num_features = prepare_mlp(X_test, X_train, y_train)

        clf = NeuralNetClassifier(
            MLPModule,
            module__input_units=num_features,
            module__output_units=num_classes,
            module__apply_softmax=True,
            module__weight_initialization=False,
            max_epochs=200,
            batch_size=min(200, len(X_train)),
            optimizer=Adam,
            optimizer__weight_decay=0.0001,
            lr=0.001,
            train_split=ValidSplit(0.1),  # Use 10% of the data for validation
            criterion=TriangularCrossEntropyLoss(num_classes=num_classes),
            callbacks=[
                EarlyStopping(patience=10, monitor="valid_loss"),
                LRScheduler(policy=ReduceLROnPlateau, patience=5, factor=0.5)
            ]
        )

    else:
        raise ValueError("Unknown Model ", model)

    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)

    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred = le.inverse_transform(y_pred)

    eval_result(y_pred, y_pred_proba, y_test, dataset, performance, le)

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
    u_ordinal_binary_variance = np.apply_along_axis(binary_ordinal_variance, 1, y_pred_proba)
    y_risk_abs = empirical_risk(y_pred_proba, "l1")
    u_absolute = risk_of_prediction(y_risk_abs, y_pred_proba, "l1")
    y_risk_squared = empirical_risk(y_pred_proba, "l2")
    u_squared = risk_of_prediction(y_risk_squared, y_pred_proba, "l2")
    random = np.random.random_sample(y_pred.shape)

    u_measures = {
        "CONF": u_confidence,
        "MARG": u_margin,
        "ENT": u_entropy,
        "VAR": u_variance,
        "$\\text{CONS}_{\\,\\text{Cns}}$": u_consensus_tastle,
        "$\\text{CONS}_{\\,C_1}$": u_consensus_leik,
        "$\\text{CONS}_{\\,C_2}$": u_consensus_blair,
        "$\\text{CONS}_{\\,C_A}$": u_agreement,
        "DFU": u_dfu,
        "$\\text{ORD}_{\\,\\text{ENT}}$": u_ordinal_binary_entropy,
        "$\\text{ORD}_{\\,\\text{MARG}}$": u_ordinal_binary_margin,
        "$\\text{ORD}_{\\,\\text{VAR}}$": u_ordinal_binary_variance,
        "$R_{l_1}$": u_absolute,
        "$R_{l_2}$": u_squared,
        "random": random
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


def prepare_mlp(X_test, X_train, y_train):
    num_features = X_train.shape[1]
    num_classes = len(set(y_train))
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    return X_test, X_train, num_classes, num_features


def eval_result(y_pred, y_pred_proba, y_test, dataset, performance, le):
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred,
                            weights="quadratic")
    print("ACC", acc)
    print("MAE", mae)
    print("MSE", mse)
    print("QWK", qwk)

    y_test = le.transform(y_test)
    o = OneHotEncoder().fit(
        np.array([i for i in range(max(len(np.unique(y_test)), y_pred_proba.shape[1]))]).reshape(-1, 1))

    brier = brier_score(o.transform(y_test.reshape(-1, 1)).toarray(), y_pred_proba)
    nll = log_loss(y_test, y_pred_proba, labels=[i for i in range(max(len(np.unique(y_test)), y_pred_proba.shape[1]))])
    ece = ECE(y_test, y_pred_proba, bins=10)

    cf_matrix = confusion_matrix(y_test, y_pred, normalize=None)
    print(cf_matrix)

    performance[dataset]["ACC"].append(acc)
    performance[dataset]["MAE"].append(mae)
    performance[dataset]["MSE"].append(mse)
    performance[dataset]["QWK"].append(qwk)
    performance[dataset]["BRIER"].append(brier)
    performance[dataset]["NLL"].append(nll)
    performance[dataset]["ECE"].append(ece)


def metric_mean_stdev_methods(overall_result_prr, metric):
    prr = np.array(overall_result_prr[metric])
    prr = prr[~np.isnan(prr).any(axis=1)]  # remove rows with nans
    prr_mean = np.mean(prr, axis=0)
    prr_std = np.std(prr, axis=0)
    return prr_mean, prr_std
