import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


class CATECombiner:

    def __init__(self, propensity_score: float, cate_learner_1, cate_learner_2):
        self.propensity_score = propensity_score
        self.cate_learner_1 = cate_learner_1
        self.cate_learner_2 = cate_learner_2
        self.lambda_val = None

    def fit(self, X, S, A, Y):

        predictions_learner_1 = np.zeros(X.shape[0])
        predictions_learner_2 = np.zeros(X.shape[0])

        for train_index, val_index in StratifiedKFold(n_splits=3, shuffle=True).split(
            X, S.squeeze() * 2 + A.squeeze()
        ):
            X_train, X_test = X[train_index], X[val_index]
            S_train, S_test = S[train_index], S[val_index]
            A_train, A_test = A[train_index], A[val_index]
            Y_train, Y_test = Y[train_index], Y[val_index]

            self.cate_learner_1.fit(X_train, S_train, A_train, Y_train)
            self.cate_learner_2.fit(X_train, S_train, A_train, Y_train)

            predictions_learner_1[val_index] = self.cate_learner_1.predict(X_test)
            predictions_learner_2[val_index] = self.cate_learner_2.predict(X_test)

        predictions_learner_1 = predictions_learner_1.reshape(-1, 1)
        predictions_learner_2 = predictions_learner_2.reshape(-1, 1)
        stacked_predictions = np.hstack([predictions_learner_1, predictions_learner_2])
        assert (
            stacked_predictions.shape[0]
            == predictions_learner_1.shape[0]
            == predictions_learner_2.shape[0]
        )

        pseudo_outcome = (
            (A / self.propensity_score - (1 - A) / (1 - self.propensity_score)) * Y
        ).reshape(-1, 1)

        # standardize predictions_learner_1 and predictions_learner_2 between 0 and 1
        min_1, max_1 = predictions_learner_1.min(), predictions_learner_1.max()
        predictions_learner_1 = (predictions_learner_1 - min_1) / (
            max_1 - min_1 + 1e-12
        )

        min_2, max_2 = predictions_learner_2.min(), predictions_learner_2.max()
        predictions_learner_2 = (predictions_learner_2 - min_2) / (
            max_2 - min_2 + 1e-12
        )

        linear_model = LinearRegression(positive=True, fit_intercept=False).fit(
            stacked_predictions,
            pseudo_outcome,
        )
        coef_normalized = linear_model.coef_ / (np.sum(linear_model.coef_) + 1e-12)
        self.lambda_val = coef_normalized[0, 0]

        # Refit learners on all data once we have determined the best combination
        self.cate_learner_1.fit(X, S, A, Y)
        self.cate_learner_2.fit(X, S, A, Y)

    def predict(self, X):
        if self.lambda_val is None:
            raise ValueError("Need to call fit() first.")
        return self.lambda_val * self.cate_learner_1.predict(X) + (
            1 - self.lambda_val
        ) * self.cate_learner_2.predict(X)

    def get_lambda(self):
        return {
            "lambda": self.lambda_val,
        }
