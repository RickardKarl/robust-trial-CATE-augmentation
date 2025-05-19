from sklearn.linear_model import LinearRegression
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
