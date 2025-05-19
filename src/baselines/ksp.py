import numpy as np
from econml.metalearners import TLearner
from sklearn.linear_model import LinearRegression


class KSPCATE:

    def __init__(
        self,
        propensity_score=None,
        cate_estimator=TLearner(models=LinearRegression()),
        bias_correction_model=LinearRegression(),
    ):

        self.propensity_score = propensity_score
        self.cate_estimator = cate_estimator
        self.bias_correction_model = bias_correction_model

    def fit(self, X, S, A, Y):

        assert X.shape[0] == S.shape[0] == A.shape[0] == Y.shape[0]  # same nbr of rows
        assert S.shape[1] == A.shape[1] == Y.shape[1] == 1

        # Learn CATE on observational data
        obs_indexing = (S == 0).squeeze()
        X_obs = X[obs_indexing, :]
        A_obs = A[obs_indexing, :].squeeze()
        Y_obs = Y[obs_indexing, :].squeeze()
        self.cate_estimator.fit(Y=Y_obs, T=A_obs, X=X_obs)

        # Learn bias correction using trial data
        trial_indexing = (S == 1).squeeze()
        X_trial = X[trial_indexing, :]
        A_trial = A[trial_indexing, :].squeeze()
        Y_trial = Y[trial_indexing, :].squeeze()

        if self.propensity_score is None:
            propensity = np.mean(A_trial)  # assume constant propensity
        else:
            propensity = self.propensity_score
        ipw_pseudo_outcome = (
            (A_trial / propensity - (1 - A_trial) / (1 - propensity)) * Y_trial
        ).squeeze()
        residual = ipw_pseudo_outcome - self.cate_estimator.effect(X_trial)
        self.bias_correction_model.fit(X_trial, residual)



    def predict(self, X):

        return self.cate_estimator.effect(X) + self.bias_correction_model.predict(X)
