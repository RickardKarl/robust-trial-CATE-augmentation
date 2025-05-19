import numpy as np
from econml.metalearners import TLearner
from sklearn.linear_model import LinearRegression


class TrialCATE:

    def __init__(
        self,
        cate_estimator=TLearner(models=LinearRegression()),
    ):

        self.cate_estimator = cate_estimator

    def get_cate_estimator(self):
        return self.cate_estimator

    def fit(self, X, S, A, Y):

        assert X.shape[0] == S.shape[0] == A.shape[0] == Y.shape[0]  # same nbr of rows
        assert S.shape[1] == A.shape[1] == Y.shape[1] == 1

        trial_index = (S == 1).squeeze()
        X = X[trial_index, :]
        A = A[trial_index, :].squeeze()
        Y = Y[trial_index, :].squeeze()

        self.cate_estimator.fit(Y=Y, T=A, X=X)

        self.ate_trial = self.cate_estimator.ate(X=X)

    def predict(self, X):

        return self.cate_estimator.effect(X)

