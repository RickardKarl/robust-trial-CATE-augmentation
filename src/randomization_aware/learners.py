from sklearn.linear_model import LogisticRegressionCV, LinearRegression
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


class AbstractRandomizationAwareLearner:

    def __init__(
        self,
        propensity_score: float,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        crossfit_folds=2,
        outcome_model=None,
    ):
        self.propensity_score = propensity_score  # assume propensity score constant 
        self.regressor_cate = regressor_cate
        self.regressor_control = regressor_control
        self.regressor_treated = regressor_treated
        self.crossfit_folds = crossfit_folds
        self.outcome_model = outcome_model

        self.fitted_cate_models = []

        self.computed_pseudo_outcomes = None

    def fit(self, X, S, A, Y):

        assert X.shape[0] == S.shape[0] == A.shape[0] == Y.shape[0]  # same nbr of rows
        assert S.shape[1] == A.shape[1] == Y.shape[1] == 1

        # Cross-fitting
        self.computed_pseudo_outcomes = np.zeros((Y.shape[0], 2))
        fold_number = 0

        if self.outcome_model is not None:
            external_index = (S == 0).squeeze()
            self.outcome_model.fit(X[external_index, :], Y[external_index].ravel())

        for train_index, test_index in StratifiedKFold(
            n_splits=self.crossfit_folds, shuffle=True
        ).split(X, S.squeeze() * 2 + A.squeeze()):
            X_train, X_test = X[train_index], X[test_index]
            S_train, S_test = S[train_index], S[test_index]
            A_train, A_test = A[train_index], A[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            predictions_control, predictions_treated = self.compute_outcome_predictions(
                X_train, S_train, A_train, Y_train, X_test
            )

            assert (
                predictions_control.shape
                == predictions_treated.shape
                == A_test.shape
                == Y_test.shape
            )

            ra_pseudo_outcome_fold = (
                (
                    A_test / self.propensity_score * (Y_test - predictions_treated)
                    - (1 - A_test)
                    / (1 - self.propensity_score)
                    * (Y_test - predictions_control)
                )
                + (predictions_treated - predictions_control)
            ).ravel()

            cate_model_fold = clone(self.regressor_cate)
            trial_index_test = (S_test == 1).squeeze()
            X_test = X_test[trial_index_test]
            if self.outcome_model is not None:
                X_test = self.augment_feature_vector(X_test)

            cate_model_fold.fit(X_test, ra_pseudo_outcome_fold[trial_index_test])
            self.fitted_cate_models.append(cate_model_fold)

            # Save for ability to retrieve computations
            self.computed_pseudo_outcomes[test_index, 0] = ra_pseudo_outcome_fold
            self.computed_pseudo_outcomes[test_index, 1] = fold_number
            fold_number += 1

    def predict(self, X):
        if len(self.fitted_cate_models) == 0:
            raise ValueError("fit must be called first")
        prediction = 0

        if self.outcome_model is not None:
            X = self.augment_feature_vector(X)

        for model in self.fitted_cate_models:
            prediction += model.predict(X)
        return prediction / len(self.fitted_cate_models)

    def augment_feature_vector(self, X):
        if self.outcome_model is not None:
            augmented_feature = self.outcome_model.predict(X).reshape(-1, 1)
            return np.column_stack([X, augmented_feature])
        else:
            raise ValueError("Need to provide outcome_model")


    def compute_outcome_predictions(self, X_train, S_train, A_train, Y_train, X_test):
        raise NotImplementedError

    def get_computed_pseudo_outcomes(self):
        if self.computed_pseudo_outcomes is None:
            raise ValueError("need to call fit first")
        return self.computed_pseudo_outcomes


class PWLearner(AbstractRandomizationAwareLearner):

    def __init__(
        self,
        propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        crossfit_folds=2,
        outcome_model=None,
    ):
        super().__init__(
            propensity_score,
            regressor_cate,
            regressor_control,
            regressor_treated,
            crossfit_folds,
            outcome_model,
        )

    def compute_outcome_predictions(self, X_train, S_train, A_train, Y_train, X_test):
        return np.zeros((X_test.shape[0],)), np.zeros((X_test.shape[0],))


class DRLearner(AbstractRandomizationAwareLearner):

    def __init__(
        self,
        propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        crossfit_folds=2,
        outcome_model=None,
    ):
        super().__init__(
            propensity_score,
            regressor_cate,
            regressor_control,
            regressor_treated,
            crossfit_folds,
            outcome_model,
        )

    def compute_outcome_predictions(self, X_train, S_train, A_train, Y_train, X_test):
        control_index_train = ((A_train == 0) & (S_train == 1)).squeeze()
        treated_index_train = ((A_train == 1) & (S_train == 1)).squeeze()

        self.regressor_control.fit(
            X_train[control_index_train, :],
            Y_train[control_index_train, :].squeeze(),
        )

        self.regressor_treated.fit(
            X_train[treated_index_train, :],
            Y_train[treated_index_train, :].squeeze(),
        )

        predictions_control = self.regressor_control.predict(X_test).reshape(-1, 1)
        predictions_treated = self.regressor_treated.predict(X_test).reshape(-1, 1)

        return predictions_control, predictions_treated


class QuasiOptimizedLearner(AbstractRandomizationAwareLearner):

    def __init__(
        self,
        propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        study_classifier=LogisticRegressionCV(),
        crossfit_folds=2,
        remove_study_weighting=False,
        outcome_model=None,
    ):
        super().__init__(
            propensity_score,
            regressor_cate,
            regressor_control,
            regressor_treated,
            crossfit_folds,
            outcome_model,
        )
        self.study_classifier = study_classifier
        self.remove_study_weighting = (
            remove_study_weighting  # to perform ablation study
        )

    def compute_outcome_predictions(self, X_train, S_train, A_train, Y_train, X_test):

        control_index_train = (A_train == 0).squeeze()
        treated_index_train = (A_train == 1).squeeze()

        self.study_classifier.fit(
            np.column_stack([X_train, A_train]), S_train.squeeze()
        )
        participant_probability_control = self.study_classifier.predict_proba(
            np.column_stack([X_train, np.zeros((X_train.shape[0], 1))])
        )[:, 1]
        participant_probability_treated = self.study_classifier.predict_proba(
            np.column_stack([X_train, np.ones((X_train.shape[0], 1))])
        )[:, 1]

        marg_participant_probability_control = np.mean(S_train[control_index_train])
        self.regressor_control.fit(
            X_train[control_index_train, :],
            Y_train[control_index_train, :].squeeze(),
            sample_weight=(
                participant_probability_control[control_index_train]
                / marg_participant_probability_control
                if not self.remove_study_weighting
                else None
            ),
        )

        marg_participant_probability_treated = np.mean(S_train[treated_index_train])
        self.regressor_treated.fit(
            X_train[treated_index_train, :],
            Y_train[treated_index_train, :].squeeze(),
            sample_weight=(
                participant_probability_treated[treated_index_train]
                / marg_participant_probability_treated
                if not self.remove_study_weighting
                else None
            ),
        )

        predictions_control = self.regressor_control.predict(X_test).reshape(-1, 1)
        predictions_treated = self.regressor_treated.predict(X_test).reshape(-1, 1)

        return predictions_control, predictions_treated


class CombinedLearner(AbstractRandomizationAwareLearner):

    """
    Combines two randomization-aware CATE learners on nuisance level.
    See combine_cate.py for learner that combines on pseudo-outcome level instead.
    """


    def __init__(
        self,
        propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        study_classifier=LogisticRegressionCV(),
        crossfit_folds=2,
        remove_study_weighting=False,
        ensembling_cv_folds=2,
        outcome_model=None,
    ):
        super().__init__(
            propensity_score,
            regressor_cate,
            regressor_control,
            regressor_treated,
            crossfit_folds,
            outcome_model,
        )
        self.study_classifier = study_classifier
        self.remove_study_weighting = remove_study_weighting

        self.ensembling_cv_folds = ensembling_cv_folds
        self.lambda_control = None
        self.lambda_treated = None

    def get_lambda(self):
        return {
            "lambda_control": self.lambda_control,
            "lambda_treated": self.lambda_treated,
        }

    def compute_outcome_predictions(self, X_train, S_train, A_train, Y_train, X_test):

        dr_learner = DRLearner(
            self.propensity_score,
            self.regressor_cate,
            self.regressor_control,
            self.regressor_treated,
            self.crossfit_folds,
        )
        quasi_optimized_learner = QuasiOptimizedLearner(
            self.propensity_score,
            self.regressor_cate,
            self.regressor_control,
            self.regressor_treated,
            self.study_classifier,
            self.crossfit_folds,
            self.remove_study_weighting,
        )

        predictions_control = np.zeros((Y_train.shape[0], 2))
        predictions_treated = np.zeros((Y_train.shape[0], 2))

        for train_index, val_index in StratifiedKFold(
            n_splits=self.ensembling_cv_folds, shuffle=True
        ).split(X_train, S_train.squeeze() * 2 + A_train.squeeze()):

            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            S_train_fold, _ = S_train[train_index], S_train[val_index]
            A_train_fold, _ = A_train[train_index], A_train[val_index]
            Y_train_fold, _ = Y_train[train_index], Y_train[val_index]

            tmp_control_dr, tmp_treated_dr = dr_learner.compute_outcome_predictions(
                X_train_fold, S_train_fold, A_train_fold, Y_train_fold, X_val_fold
            )
            tmp_control_quasi, tmp_treated_quasi = (
                quasi_optimized_learner.compute_outcome_predictions(
                    X_train_fold, S_train_fold, A_train_fold, Y_train_fold, X_val_fold
                )
            )

            predictions_control[val_index, 0] = tmp_control_dr.ravel()
            predictions_treated[val_index, 0] = tmp_treated_dr.ravel()
            predictions_control[val_index, 1] = tmp_control_quasi.ravel()
            predictions_treated[val_index, 1] = tmp_treated_quasi.ravel()

        # Find lambda_control and lambda_treated

        control_index = (A_train == 0).ravel()
        model_control = LinearRegression(positive=True, fit_intercept=False).fit(
            predictions_control[control_index, :],
            Y_train[control_index],
        )
        model_control_coef_normalized = model_control.coef_ / np.sum(
            model_control.coef_
        )
        lambda_control = model_control_coef_normalized[0, 0]

        treated_index = (A_train == 1).ravel()
        model_treated = LinearRegression(positive=True, fit_intercept=False).fit(
            predictions_treated[treated_index, :], Y_train[treated_index]
        )
        model_treated_coef_normalized = model_treated.coef_ / np.sum(
            model_treated.coef_
        )
        lambda_treated = model_treated_coef_normalized[0, 0]

        assert (
            np.abs(np.sum(model_control_coef_normalized) - 1) < 1e-9
            and np.abs(np.sum(model_treated_coef_normalized) - 1) < 1e-9
        )

        # Compute final predictions
        dr_predictions_control, dr_predictions_treated = (
            dr_learner.compute_outcome_predictions(
                X_train, S_train, A_train, Y_train, X_test
            )
        )
        quasi_predictions_control, quasi_predictions_treated = (
            quasi_optimized_learner.compute_outcome_predictions(
                X_train, S_train, A_train, Y_train, X_test
            )
        )

        predictions_control = (
            lambda_control * dr_predictions_control
            + (1 - lambda_control) * quasi_predictions_control
        )

        predictions_treated = (
            lambda_treated * dr_predictions_treated
            + (1 - lambda_treated) * quasi_predictions_treated
        )

        self.lambda_control = lambda_control
        self.lambda_treated = lambda_treated

        return predictions_control, predictions_treated
