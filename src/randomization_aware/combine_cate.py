import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


class CATECombiner:

    def __init__(
        self, propensity_score: float, cate_learner_1, cate_learner_2, n_splits_cv=5
    ):
        self.propensity_score = propensity_score
        self.cate_learner_1 = cate_learner_1
        self.cate_learner_2 = cate_learner_2
        self.weights = None
        self.n_splits_cv = n_splits_cv

    def fit(self, X, S, A, Y):

        n = X.shape[0]
        learners = [self.cate_learner_1, self.cate_learner_2]

        # Collect out-of-fold predictions for each learner
        oof_predictions = np.zeros((n, len(learners)))

        for train_index, val_index in StratifiedKFold(
            n_splits=self.n_splits_cv, shuffle=True
        ).split(X, np.ravel(S) * 2 + np.ravel(A)):
            X_train, X_val = X[train_index], X[val_index]
            S_train, _ = S[train_index], S[val_index]
            A_train, _ = A[train_index], A[val_index]
            Y_train, _ = Y[train_index], Y[val_index]

            for j, learner in enumerate(learners):
                learner.fit(X_train, S_train, A_train, Y_train)
                oof_predictions[val_index, j] = learner.predict(X_val)

        # Build pseudo-outcome
        pseudo_outcome = (
            (A / self.propensity_score - (1 - A) / (1 - self.propensity_score))
            * (Y - np.mean(Y[(S == 1).ravel()]))
        ).reshape(-1, 1)

        # Fit linear combination model (non-negative, no intercept)
        linear_model = LinearRegression(positive=True, fit_intercept=False).fit(
            oof_predictions,
            pseudo_outcome,
        )
        coefs = linear_model.coef_.reshape(-1)
        self.weights = coefs / (np.sum(coefs) + 1e-12)

        # Refit learners on full dataset
        for learner in learners:
            learner.fit(X, S, A, Y)

    def predict(self, X):
        if not hasattr(self, "weights"):
            raise ValueError("Need to call fit() first.")
        learners = [self.cate_learner_1, self.cate_learner_2]
        preds = np.column_stack([learner.predict(X) for learner in learners])
        return preds @ self.weights  # weighted combination

    def get_lambda(self):
        return {"weights": self.weights}
