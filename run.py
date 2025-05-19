from itertools import product
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

import numpy as np
import pandas as pd

from src.data.synthetic import Synthetic
from methods import get_method
from src.utils import compute_area_under_qini_curve
import random

import os

if not os.path.exists("results"):
    os.makedirs("results")

####################################
# Define available methods
available_methods = [
    "DRLearnerLinear",
    "QuasiOptimizedLinear",
    "CombinedNuiLinear",
    "CombinedPsOLinear",
    "AsaieeLinear",
    "TLearnerLinear",
    "KSPLinear_T",
    "PoolingLinear_T",
    "KSPLinear_DR",
    "DRLearnerMixed",
    "QuasiOptimizedMixed",
    "CombinedNuiMixed",
    "CombinedPsOMixed",
    "AsaieeMixed",
    "KSPMixed_DR",
    "DRLearnerNonlinear",
    "QuasiOptimizedNonlinear",
    "CombinedNuiNonlinear",
    "CombinedPsONonlinear",
    "AsaieeNonlinear",
    "TLearnerNonlinear",
    "PoolingNonlinear_T",
    "KSPNonlinear_T",
    "KSPNonlinear_DR",
    "DM",
]

# Assert that all of the above methods do not lead to raising an error when calling get_method
for method_name in available_methods:
    try:
        method = get_method(method_name)
        assert method is not None, f"get_method returned None for {method_name}"
    except Exception as e:
        raise RuntimeError(f"Error occurred while getting method {method_name}: {e}")

#####################################
# Read arguments


parser = argparse.ArgumentParser(description="Run synthetic data experiments.")
parser.add_argument(
    "--methods",
    nargs="+",
    choices=available_methods,
    default=available_methods,
    help="List of methods",
)

parser.add_argument("--seed", type=int, default=None, help="Set seed for experiment")
parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
parser.add_argument(
    "--n_trial",
    type=int,
    nargs="+",
    default=[100],
    help="List of trial sizes",
)
parser.add_argument(
    "--n_obs",
    type=int,
    nargs="+",
    default=[1000],
    help="List of observation sizes",
)
parser.add_argument(
    "--n_features", type=int, nargs="+", default=[5], help="List of feature sizes"
)
parser.add_argument(
    "--misspecification",
    type=str,
    nargs="+",
    default=["False", "True"],
    help="Determine if misspecification to DGP is occuring",
)

parser.add_argument(
    "--n_crossfit_folds",
    type=int,
    default=2,
    help="Number of folds for crossfitting",
)
parser.add_argument(
    "--n_jobs",
    type=int,
    default=1,
    help="Number of threads to use for parallel processing",
)


args = parser.parse_args()
methods = args.methods
seed = args.seed
iterations = args.iterations
n_trial_list = args.n_trial
n_obs_list = args.n_obs
n_features_list = args.n_features
misspecification_list = [m.lower() == "true" for m in args.misspecification]
n_crossfit_folds = args.n_crossfit_folds
n_jobs = args.n_jobs


configs = list(
    product(n_trial_list, n_obs_list, n_features_list, misspecification_list)
)
np.random.seed(seed)
timestamp = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")

filename = f"{timestamp}_{'_'.join(methods)}_{seed}_{random.randint(1000, 9999)}"
print(f"Results will be saved in {filename}.csv")


#########################################
# Start experimental loop
#########################################


def run_experiment(n_trial, n_obs, n_features, misspecification, i):
    if misspecification:
        dgp = Synthetic(population_shift=0.2, n_features=n_features + 2)
        covariates = dgp.get_covar()
        covariates = covariates[:-2]
    else:
        dgp = Synthetic(population_shift=0.2, n_features=n_features)
        covariates = dgp.get_covar()

    data_train = dgp.sample(n_trial=n_trial, n_obs=n_obs)
    data_test = dgp.sample(n_trial=50_000, n_obs=2)
    data_test = data_test.loc[data_test.S == 1]

    experiment_metrics = []

    for method_name in methods:
        estimator = get_method(method_name, n_crossfit_folds)

        estimator.fit(
            data_train[covariates].values,
            data_train["S"].values.reshape(-1, 1),
            data_train["A"].values.reshape(-1, 1),
            data_train["Y"].values.reshape(-1, 1),
        )

        predictions = estimator.predict(data_test[covariates].values)
        true_cate = data_test["cate"].values

        assert (
            predictions.shape == true_cate.shape
        ), f"shapes not matching: {predictions.shape} != {true_cate.shape}"

        bias = np.mean(predictions - true_cate)
        rmse = np.sqrt(np.mean((predictions - true_cate) ** 2))
        area_under_qini_curve = compute_area_under_qini_curve(data_test, predictions)

        try:
            best_lambda = estimator.get_lambda()
        except AttributeError:
            best_lambda = {"lambda_control": np.NaN, "lambda_treated": np.NaN}

        metrics = {
            "iter": i,
            "seed": seed,
            "n_trial": n_trial,
            "n_obs": n_obs,
            "n_features": n_features,
            "misspecification": misspecification,
            "method": method_name,
            "bias": bias,
            "rmse": rmse,
            "auc": area_under_qini_curve,
            "n_crossfit_folds": n_crossfit_folds,
            **best_lambda,
        }

        experiment_metrics.append(metrics)

    return experiment_metrics


with ThreadPoolExecutor(max_workers=n_jobs) as executor:
    futures = []
    results = []
    for n_trial, n_obs, n_features, misspecification in tqdm(
        configs, desc="Config combinations"
    ):
        for i in range(iterations):
            futures.append(
                executor.submit(
                    run_experiment, n_trial, n_obs, n_features, misspecification, i
                )
            )

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing iterations",
    ):
        try:
            metrics = future.result()
            if isinstance(metrics, list):
                results.extend(metrics)
            else:
                results.append(metrics)

        except Exception as e:
            print(f"error: {str(e)}, traceback: {traceback.format_exc()}")
            results.append({"error": str(e), "traceback": traceback.format_exc()})

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"results/{filename}.csv", index=False)

print(f"Results saved to results/{filename}.csv")
