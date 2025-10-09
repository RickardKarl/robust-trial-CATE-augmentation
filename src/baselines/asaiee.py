from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
import numpy as np

from src.randomization_aware.learners import AbstractRandomizationAwareLearner


class AsaieeCATE(AbstractRandomizationAwareLearner):

    def __init__(
        self,
        propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        crossfit_folds=2,
    ):
        super().__init__(
            propensity_score,
            regressor_cate,
            regressor_control,
            regressor_treated,
            crossfit_folds,
        )

    def compute_outcome_predictions(self, X_train, S_train, A_train, Y_train, X_test):

        control_index_train = ((A_train == 0) & (S_train == 0)).squeeze()
        treated_index_train = ((A_train == 1) & (S_train == 0)).squeeze()

        self.regressor_control.fit(
            X_train[control_index_train, :], Y_train[control_index_train, :].squeeze()
        )

        self.regressor_treated.fit(
            X_train[treated_index_train, :], Y_train[treated_index_train, :].squeeze()
        )

        predictions_control = self.regressor_control.predict(X_test).reshape(-1, 1)
        predictions_treated = self.regressor_treated.predict(X_test).reshape(-1, 1)

        # we return P(A=1|X,S=1) * E[Y|X,A=0,S=0] + P(A=0|X,S=1) * E[Y|X,A=1,S=0] (no typo!)
        predictions = (
            self.propensity_score * predictions_control
            + (1 - self.propensity_score) * predictions_treated
        )
        return predictions, predictions


class OSCARCATE:
    def __init__(
        self,
        propensity_score,
        regressor_control: BaseEstimator = LinearRegression(),
        regressor_treated: BaseEstimator = LinearRegression(),
        bias_function: BaseEstimator = LassoCV(),
    ):
        """
        Bias-calibrated variant of AsaieeCATE
        """
        self.propensity_score = propensity_score
        self.regressor_control = regressor_control
        self.regressor_treated = regressor_treated
        self.bias_control = clone(bias_function)
        self.bias_treated = clone(bias_function)

    def fit(self, X, S, A, Y):

        control_index_train_obs = ((A == 0) & (S == 0)).squeeze()
        treated_index_train_obs = ((A == 1) & (S == 0)).squeeze()

        self.regressor_control.fit(
            X[control_index_train_obs, :],
            Y[control_index_train_obs, :].squeeze(),
        )

        self.regressor_treated.fit(
            X[treated_index_train_obs, :],
            Y[treated_index_train_obs, :].squeeze(),
        )

        predictions_control = self.regressor_control.predict(X).reshape(-1, 1)
        predictions_treated = self.regressor_treated.predict(X).reshape(-1, 1)

        assert predictions_control.shape == Y.shape
        assert predictions_treated.shape == Y.shape

        ########### Bias calibration ###########

        control_index_train_rct = ((A == 0) & (S == 0)).squeeze()
        treated_index_train_rct = ((A == 1) & (S == 0)).squeeze()

        propensity_ratio = (1 - self.propensity_score) / self.propensity_score

        pseudo_outcome_control = (
            Y / (1 - self.propensity_score)
            - (1 + 1 / propensity_ratio) * predictions_control
        ) / -(1 + 1 / propensity_ratio)
        pseudo_outcome_treated = (
            Y / self.propensity_score - (1 + propensity_ratio) * predictions_treated
        ) / (1 + propensity_ratio)

        self.bias_control.fit(
            X[control_index_train_rct, :],
            pseudo_outcome_control[control_index_train_rct, :].squeeze(),
        )
        self.bias_treated.fit(
            X[treated_index_train_rct, :],
            pseudo_outcome_treated[treated_index_train_rct, :].squeeze(),
        )

    def predict(self, X):

        predictions_control = self.regressor_control.predict(X)
        predictions_treated = self.regressor_treated.predict(X)

        bias_control = self.bias_control.predict(X)
        bias_treated = self.bias_treated.predict(X)

        return (predictions_treated + bias_treated) - (
            predictions_control - bias_control
        )


class ROSCARCATE:
    def __init__(
        self,
        propensity_score,
        regressor_control: BaseEstimator = LinearRegression(),
        regressor_treated: BaseEstimator = LinearRegression(),
        bias_function_outcome: BaseEstimator = LassoCV(),
        bias_function_cate: BaseEstimator = LassoCV(),
        crossfit_folds=2,
    ):
        """
        Bias-calibrated variant of AsaieeCATE
        """
        self.propensity_score = propensity_score
        self.regressor_control = regressor_control
        self.regressor_treated = regressor_treated
        self.bias_control_outcome = clone(bias_function_outcome)
        self.bias_treated_outcome = clone(bias_function_outcome)
        self.bias_cate = clone(bias_function_cate)
        self.crossfit_folds = crossfit_folds

        # to save functions over cross-fitted folds
        self.fitted_bias_control_outcome = []
        self.fitted_bias_treated_outcome = []
        self.fitted_bias_cate = []

    def fit(self, X, S, A, Y):

        control_index_obs = ((A == 0) & (S == 0)).squeeze()
        treated_index_obs = ((A == 1) & (S == 0)).squeeze()

        self.regressor_control.fit(
            X[control_index_obs, :],
            Y[control_index_obs, :].squeeze(),
        )

        self.regressor_treated.fit(
            X[treated_index_obs, :],
            Y[treated_index_obs, :].squeeze(),
        )

        predictions_control = self.regressor_control.predict(X).reshape(-1, 1)
        predictions_treated = self.regressor_treated.predict(X).reshape(-1, 1)

        assert predictions_control.shape == Y.shape
        assert predictions_treated.shape == Y.shape

        index_rct = (S == 1).squeeze()
        X_rct = X[index_rct, :]
        A_rct = A[index_rct, :]
        Y_rct = Y[index_rct, :]
        predictions_control_rct = predictions_control[index_rct, :]
        predictions_treated_rct = predictions_treated[index_rct, :]

        for train_index, test_index in StratifiedKFold(
            n_splits=self.crossfit_folds, shuffle=True
        ).split(X_rct, A_rct.squeeze()):

            X_rct_train, X_rct_test = X_rct[train_index], X_rct[test_index]
            A_rct_train, A_rct_test = A_rct[train_index], A_rct[test_index]
            Y_rct_train, Y_rct_test = Y_rct[train_index], Y_rct[test_index]
            predictions_control_rct_train, predictions_control_rct_test = (
                predictions_control_rct[train_index],
                predictions_control_rct[test_index],
            )
            predictions_treated_rct_train, predictions_treated_rct_test = (
                predictions_treated_rct[train_index],
                predictions_treated_rct[test_index],
            )

            ########### Bias calibration of outcome ###########

            control_index_train_rct = (A_rct_train == 0).squeeze()
            treated_index_train_rct = (A_rct_train == 1).squeeze()

            bias_control_outcome_model = clone(self.bias_control_outcome)
            bias_treated_outcome_model = clone(self.bias_treated_outcome)

            bias_control_outcome_model.fit(
                X_rct_train[control_index_train_rct, :],
                (
                    Y_rct_train[control_index_train_rct, :]
                    - predictions_control_rct_train[control_index_train_rct, :]
                ).squeeze(),
            )
            bias_treated_outcome_model.fit(
                X_rct_train[treated_index_train_rct, :],
                (
                    Y_rct_train[treated_index_train_rct, :]
                    - predictions_treated_rct_train[treated_index_train_rct, :]
                ).squeeze(),
            )

            bias_correction_control = bias_control_outcome_model.predict(
                X_rct_test
            ).reshape(-1, 1)
            bias_correction_treated = bias_treated_outcome_model.predict(
                X_rct_test
            ).reshape(-1, 1)

            assert (
                bias_correction_control.shape
                == Y_rct_test.shape
                == predictions_control_rct_test.shape
            )
            assert (
                bias_correction_treated.shape
                == Y_rct_test.shape
                == predictions_treated_rct_test.shape
            )

            self.fitted_bias_control_outcome.append(bias_control_outcome_model)
            self.fitted_bias_treated_outcome.append(bias_treated_outcome_model)

            ########### Bias calibration of CATE ###########

            corrected_prediction_control = (
                predictions_control_rct_test + bias_correction_control
            )
            corrected_prediction_treated = (
                predictions_treated_rct_test + bias_correction_treated
            )
            cmo = (
                self.propensity_score * predictions_control_rct_test
                + (1 - self.propensity_score) * predictions_treated_rct_test
            )

            pseudo_outcome = (
                A_rct_test / self.propensity_score
                - (1 - A_rct_test) / (1 - self.propensity_score)
            ) * (Y_rct_test - cmo)
            corrected_cate = corrected_prediction_treated - corrected_prediction_control

            bias_cate_model = clone(self.bias_cate)
            bias_cate_model.fit(
                X_rct_test,
                (pseudo_outcome - corrected_cate).squeeze(),
            )

            self.fitted_bias_cate.append(bias_cate_model)

    def predict(self, X):

        predictions_control = self.regressor_control.predict(X)
        predictions_treated = self.regressor_treated.predict(X)

        bias_control = np.mean(
            [
                fitted_model.predict(X)
                for fitted_model in self.fitted_bias_control_outcome
            ],
            axis=0,
        )
        bias_treated = np.mean(
            [
                fitted_model.predict(X)
                for fitted_model in self.fitted_bias_treated_outcome
            ],
            axis=0,
        )

        bias_cate = np.mean(
            [fitted_model.predict(X) for fitted_model in self.fitted_bias_cate], axis=0
        )

        return (
            (predictions_treated + bias_treated)
            - (predictions_control + bias_control)
            + bias_cate
        )
