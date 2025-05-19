import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, LogisticRegressionCV


class TLearnerPooling:

    def __init__(
        self,
        regressor_control: BaseEstimator = LinearRegression(),
        regressor_treated: BaseEstimator = LinearRegression(),
        study_classifier: BaseEstimator = LogisticRegressionCV(),
    ):

        self.regressor_control = regressor_control
        self.regressor_treated = regressor_treated
        self.study_classifier = study_classifier

    def fit(self, X, S, A, Y):

        assert X.shape[0] == S.shape[0] == A.shape[0] == Y.shape[0]  # same nbr of rows
        assert S.shape[1] == A.shape[1] == Y.shape[1] == 1

        # Fit separate models for S=0 and S=1
        # Calculate sample weights based on the probability of being in S=1 given X

        # Fit a logistic regression model to estimate P(S=1 | X)
        self.study_classifier.fit(X, S.ravel())
        prob_s1 = self.study_classifier.predict_proba(X)[:, 1]  # Probability of S=1

        # Assign sample weights based on the estimated probabilities
        sample_weight_control = prob_s1[(A == 0).squeeze()]
        sample_weight_treated = prob_s1[(A == 1).squeeze()]

        # Fit models with sample weights
        self.regressor_control.fit(
            X[(A == 0).squeeze(), :],
            Y[(A == 0).squeeze()].ravel(),
            sample_weight=sample_weight_control,
        )
        self.regressor_treated.fit(
            X[(A == 1).squeeze(), :],
            Y[(A == 1).squeeze()].ravel(),
            sample_weight=sample_weight_treated,
        )

    def predict(self, X):

        return self.regressor_treated.predict(X) - self.regressor_control.predict(X)
