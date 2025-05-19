from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingRegressor


from econml.metalearners import TLearner
from econml.dr import DRLearner as econml_DRLearner


from src.randomization_aware.learners import (
    DRLearner,
    QuasiOptimizedLearner,
    CombinedLearner,
)
from src.randomization_aware.combine_cate import CATECombiner
from src.baselines.asaiee import AsaieeCATE
from src.baselines.trial_only import TrialCATE
from src.baselines.pooling import TLearnerPooling
from src.baselines.ksp import KSPCATE
from src.baselines.predict_ate import DifferenceInMeans


propensity_score = 1 / 2
nonlinear_regressor = HistGradientBoostingRegressor

linear_tlearner = lambda: TLearner(models=LinearRegression())
linear_drlearner = lambda n_folds: econml_DRLearner(
    model_propensity=LogisticRegressionCV(),
    model_regression=LinearRegression(),
    model_final=LinearRegression(),
    cv=n_folds,
)

mixed_drlearner = lambda n_folds: econml_DRLearner(
    model_propensity=LogisticRegressionCV(),
    model_regression=nonlinear_regressor(),
    model_final=LinearRegression(),
    cv=n_folds,
)


nonlinear_tlearner = lambda: TLearner(models=nonlinear_regressor())
nonlinear_drlearner = lambda n_folds: econml_DRLearner(
    model_propensity=LogisticRegressionCV(),
    model_regression=nonlinear_regressor(),
    model_final=nonlinear_regressor(),
    cv=n_folds,
)

########################
# Models with parametric CATE model and nuisance models
########################


def get_pooling_linear_t(n_crossfit_folds=None):
    return TLearnerPooling(
        regressor_treated=LinearRegression(), regressor_control=LinearRegression()
    )


def get_ksp_linear_t(n_crossfit_folds=None):
    return KSPCATE(cate_estimator=linear_tlearner())


def get_ksp_linear_dr(n_crossfit_folds):
    return KSPCATE(cate_estimator=linear_drlearner(n_crossfit_folds))


def get_tlinear_linear(n_crossfit_folds=None):
    return TrialCATE(cate_estimator=linear_tlearner())


def get_drlearner_linear(n_crossfit_folds):
    return DRLearner(
        propensity_score=propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        crossfit_folds=n_crossfit_folds,
    )


def get_quasioptimized_linear(n_crossfit_folds):
    return QuasiOptimizedLearner(
        propensity_score=propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        study_classifier=LogisticRegressionCV(),
        crossfit_folds=n_crossfit_folds,
    )


def get_combined_nuisance_linear(n_crossfit_folds):
    return CombinedLearner(
        propensity_score=propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        study_classifier=LogisticRegressionCV(),
        crossfit_folds=n_crossfit_folds,
    )


def get_combined_pseudo_linear(n_crossfit_folds):
    return CATECombiner(
        propensity_score=propensity_score,
        cate_learner_1=get_drlearner_linear(n_crossfit_folds=n_crossfit_folds),
        cate_learner_2=get_quasioptimized_linear(n_crossfit_folds=n_crossfit_folds),
    )


def get_asaiee_linear(n_crossfit_folds):
    return AsaieeCATE(
        propensity_score=propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=LinearRegression(),
        regressor_treated=LinearRegression(),
        crossfit_folds=n_crossfit_folds,
    )


########################
# Models with parametric CATE model and nonparametric nuisance models
########################


def get_ksp_mixed_dr(n_crossfit_folds):
    return KSPCATE(cate_estimator=mixed_drlearner(n_crossfit_folds))


def get_drlearner_mixed(n_crossfit_folds):
    return TrialCATE(cate_estimator=mixed_drlearner(n_crossfit_folds))


def get_quasioptimized_mixed(n_crossfit_folds):
    return QuasiOptimizedLearner(
        propensity_score=propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=nonlinear_regressor(),
        regressor_treated=nonlinear_regressor(),
        study_classifier=LogisticRegressionCV(),
        crossfit_folds=n_crossfit_folds,
    )


def get_combined_nuisance_mixed(n_crossfit_folds):
    return CombinedLearner(
        propensity_score=propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=nonlinear_regressor(),
        regressor_treated=nonlinear_regressor(),
        study_classifier=LogisticRegressionCV(),
        crossfit_folds=n_crossfit_folds,
    )


def get_combined_pseudo_mixed(n_crossfit_folds):
    return CATECombiner(
        propensity_score=1 / 2,
        cate_learner_1=get_drlearner_mixed(n_crossfit_folds),
        cate_learner_2=get_quasioptimized_mixed(n_crossfit_folds),
    )


def get_asaiee_mixed(n_crossfit_folds):
    return AsaieeCATE(
        propensity_score=propensity_score,
        regressor_cate=LinearRegression(),
        regressor_control=nonlinear_regressor(),
        regressor_treated=nonlinear_regressor(),
        crossfit_folds=n_crossfit_folds,
    )


########################
# Models with nonparametric CATE model and nuisance models
########################


def get_pooling_nonlinear_t(n_crossfit_folds=None):
    return TLearnerPooling(
        regressor_control=nonlinear_regressor(), regressor_treated=nonlinear_regressor()
    )


def get_ksp_nonlinear_t(n_crossfit_folds=None):
    return KSPCATE(cate_estimator=nonlinear_tlearner())


def get_ksp_nonlinear_dr(n_crossfit_folds):
    return KSPCATE(cate_estimator=nonlinear_drlearner(n_crossfit_folds))


def get_tlearner_nonlinear(n_crossfit_folds=None):
    return TrialCATE(cate_estimator=nonlinear_tlearner())


def get_drlearner_nonlinear(n_crossfit_folds):
    return TrialCATE(cate_estimator=nonlinear_drlearner(n_crossfit_folds))


def get_quasioptimized_nonlinear(n_crossfit_folds):
    return QuasiOptimizedLearner(
        propensity_score=propensity_score,
        regressor_cate=nonlinear_regressor(),
        regressor_control=nonlinear_regressor(),
        regressor_treated=nonlinear_regressor(),
        study_classifier=LogisticRegressionCV(),
        crossfit_folds=n_crossfit_folds,
    )


def get_combined_nuisance_nonlinear(n_crossfit_folds):
    return CombinedLearner(
        propensity_score=propensity_score,
        regressor_cate=nonlinear_regressor(),
        regressor_control=nonlinear_regressor(),
        regressor_treated=nonlinear_regressor(),
        study_classifier=LogisticRegressionCV(),
        crossfit_folds=n_crossfit_folds,
    )


def get_combined_pseudo_nonlinear(n_crossfit_folds):
    return CATECombiner(
        propensity_score=1 / 2,
        cate_learner_1=get_drlearner_nonlinear(n_crossfit_folds),
        cate_learner_2=get_quasioptimized_nonlinear(n_crossfit_folds),
    )


def get_asaiee_nonlinear(n_crossfit_folds):
    return AsaieeCATE(
        propensity_score=propensity_score,
        regressor_cate=nonlinear_regressor(),
        regressor_control=nonlinear_regressor(),
        regressor_treated=nonlinear_regressor(),
        crossfit_folds=n_crossfit_folds,
    )


########################
# Other baselines
########################


def get_dm(n_crossfit_folds=None):
    return DifferenceInMeans()


######################################################
# Function to call above functins
######################################################


def get_method(method_name: str, n_crossfit_folds=None):

    if method_name == "DRLearnerLinear":
        return get_drlearner_linear(n_crossfit_folds)
    elif method_name == "QuasiOptimizedLinear":
        return get_quasioptimized_linear(n_crossfit_folds)
    elif method_name == "CombinedNuiLinear":
        return get_combined_nuisance_linear(n_crossfit_folds)
    elif method_name == "CombinedPsOLinear":
        return get_combined_pseudo_linear(n_crossfit_folds)
    elif method_name == "AsaieeLinear":
        return get_asaiee_linear(n_crossfit_folds)
    elif method_name == "TLearnerLinear":
        return get_tlinear_linear(n_crossfit_folds)
    elif method_name == "PoolingLinear_T":
        return get_pooling_linear_t(n_crossfit_folds)
    elif method_name == "KSPLinear_T":
        return get_ksp_linear_t(n_crossfit_folds)
    elif method_name == "KSPLinear_DR":
        return get_ksp_linear_dr(n_crossfit_folds)

    elif method_name == "DRLearnerMixed":
        return get_drlearner_mixed(n_crossfit_folds)
    elif method_name == "QuasiOptimizedMixed":
        return get_quasioptimized_mixed(n_crossfit_folds)
    elif method_name == "CombinedNuiMixed":
        return get_combined_nuisance_mixed(n_crossfit_folds)
    elif method_name == "CombinedPsOMixed":
        return get_combined_pseudo_mixed(n_crossfit_folds)
    elif method_name == "AsaieeMixed":
        return get_asaiee_mixed(n_crossfit_folds)
    elif method_name == "KSPMixed_DR":
        return get_ksp_mixed_dr(n_crossfit_folds)

    elif method_name == "DRLearnerNonlinear":
        return get_drlearner_nonlinear(n_crossfit_folds)
    elif method_name == "QuasiOptimizedNonlinear":
        return get_quasioptimized_nonlinear(n_crossfit_folds)
    elif method_name == "CombinedNuiNonlinear":
        return get_combined_nuisance_nonlinear(n_crossfit_folds)
    elif method_name == "CombinedPsONonlinear":
        return get_combined_pseudo_nonlinear(n_crossfit_folds)
    elif method_name == "AsaieeNonlinear":
        return get_asaiee_nonlinear(n_crossfit_folds)
    elif method_name == "TLearnerNonlinear":
        return get_tlearner_nonlinear(n_crossfit_folds)
    elif method_name == "PoolingNonlinear_T":
        return get_pooling_nonlinear_t(n_crossfit_folds)
    elif method_name == "KSPNonlinear_T":
        return get_ksp_nonlinear_t(n_crossfit_folds)
    elif method_name == "KSPNonlinear_DR":
        return get_ksp_nonlinear_dr(n_crossfit_folds)
    elif method_name == "DM":
        return get_dm(n_crossfit_folds)
    else:
        raise ValueError(f"method {method_name} not found")
