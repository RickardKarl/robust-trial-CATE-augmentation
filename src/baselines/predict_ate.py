import numpy as np


class DifferenceInMeans:

    def __init__(self):
        self.ate_trial = None

    def fit(self, X, S, A, Y):
        mean_Y1 = np.mean(Y[((A == 1) & (S == 1)).ravel()])
        mean_Y0 = np.mean(Y[((A == 0) & (S == 1)).ravel()])
        self.ate_trial = mean_Y1 - mean_Y0

    def predict(self, X):
        ate_trial = self.get_ate_trial()
        return ate_trial * np.ones((X.shape[0],))

    def get_ate_trial(self):
        if self.ate_trial is None:
            raise ValueError("need to call fit first")
        return self.ate_trial
