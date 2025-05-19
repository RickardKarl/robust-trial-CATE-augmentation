import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
Based on processing of STAR dataset by Kallus et al. (2018)
"""


def _to_categorical(X):
    assert type(X) == type(np.ones((1,)))
    return pd.get_dummies(pd.DataFrame(X.astype(str)), dummy_na=True).values.astype(int)


def preprocess_star_dataset(star_data: pd.DataFrame, cat_covar_columns: list):

    # Following Kallus et al.
    # _confounding_covar = "g1surban"
    Y_columns = ["g1treadss", "g1tmathss", "g1tlistss"]

    star_data.drop(columns=["Unnamed: 43", "Unnamed: 44"], inplace=True)

    treatment_filter = np.isfinite(star_data.g1classtype)
    outcome_filter = np.isfinite(
        star_data.g1tlistss + star_data.g1treadss + star_data.g1tmathss
    )

    T_all = star_data.g1classtype[
        np.logical_and(treatment_filter, outcome_filter)
    ].values
    X_all = _to_categorical(
        star_data[cat_covar_columns][np.logical_and(treatment_filter, outcome_filter)]
        .fillna(0)
        .values[T_all != 3]
    )
    Y_cols = star_data[Y_columns][
        np.logical_and(treatment_filter, outcome_filter)
    ].values[T_all != 3]
    urban = star_data.g1surban[np.logical_and(outcome_filter, treatment_filter)].values[
        T_all != 3
    ]
    rural_filter = np.logical_or(urban == 1, urban == 3)  # 1 = inner city, 3 = rural
    urban_filter = np.logical_or(urban == 2, urban == 4)  # 2 = suburban, 4 = urban

    T_all = star_data.g1classtype[
        np.logical_and(treatment_filter, outcome_filter)
    ].values[T_all != 3]
    T_all[T_all == 2] = 0
    Y_all = np.sum(Y_cols, axis=1) / 3

    X_rural = X_all[rural_filter]
    X_urban = X_all[urban_filter]

    T_rural = T_all[rural_filter]
    T_urban = T_all[urban_filter]

    Y_rural = Y_all[rural_filter]
    Y_urban = Y_all[urban_filter]

    _propensity_score = np.mean(T_all)
    _mean_outcome = np.mean(Y_rural)

    return (
        X_rural,
        X_urban,
        T_rural,
        T_urban,
        Y_rural,
        Y_urban,
        _propensity_score,
        _mean_outcome,
    )


def _generate_rct_obs(
    X_rural,
    Y_rural,
    T_rural,
    X_urban,
    Y_urban,
    T_urban,
    rct_fraction_of_rural=0.5,
    eval_fraction_of_rct=0.4,
):

    X_urban_control = X_urban[T_urban == 0]
    X_urban_treated = X_urban[T_urban == 1]

    Y_urban_control = Y_urban[T_urban == 0]
    Y_urban_treated = Y_urban[T_urban == 1]

    X_rct, X_rural_not_RCT, Y_rct, Y_rural_not_RCT, T_rct, T_rural_not_rct = (
        train_test_split(X_rural, Y_rural, T_rural, test_size=1 - rct_fraction_of_rural)
    )

    (
        X_rct,
        X_rct_eval,
        Y_rct,
        Y_rct_eval,
        T_rct,
        T_rct_eval,
    ) = train_test_split(
        X_rct,
        Y_rct,
        T_rct,
        test_size=eval_fraction_of_rct,
    )

    X_notrct = np.vstack((X_rural_not_RCT, X_urban))
    Y_notrct = np.hstack((Y_rural_not_RCT, Y_urban))
    T_notrct = np.hstack((T_rural_not_rct, T_urban))
    _local_rural_filter = np.array(
        [True] * X_rural_not_RCT.shape[0] + [False] * X_urban.shape[0]
    ).ravel()
    assert _local_rural_filter.shape == Y_notrct.shape

    X_notrct_rural = X_notrct[_local_rural_filter]
    Y_notrct_rural = Y_notrct[_local_rural_filter]
    T_notrct_rural = T_notrct[_local_rural_filter]

    # checked
    X_notrct_rural_treated = X_notrct_rural[T_notrct_rural == 1]
    X_notrct_rural_control = X_notrct_rural[T_notrct_rural == 0]

    # checked
    Y_notrct_rural_treated = Y_notrct_rural[T_notrct_rural == 1]
    Y_notrct_rural_control = Y_notrct_rural[T_notrct_rural == 0]

    # OUTCOME BASED REMOVAL
    _treated_rural_filter = Y_notrct_rural_treated < np.median(Y_notrct_rural_treated)
    _treated_urban_filter = Y_urban_treated < np.median(Y_urban[T_urban == 1])

    # remove large samples.
    X_obs_treated_rural = X_notrct_rural_treated[_treated_rural_filter, :]
    Y_obs_treated_rural = Y_notrct_rural_treated[_treated_rural_filter]

    X_obs_treated_urban = X_urban_treated[_treated_urban_filter, :]
    Y_obs_treated_urban = Y_urban_treated[_treated_urban_filter]

    assert X_obs_treated_rural.shape[0] == Y_obs_treated_rural.shape[0]

    # checked
    X_obs = np.vstack(
        (
            X_obs_treated_rural,
            X_obs_treated_urban,
            X_notrct_rural_control,
            X_urban_control,
        )
    )

    # checked
    Y_obs = np.vstack(
        (
            Y_obs_treated_rural.reshape(-1, 1),
            Y_obs_treated_urban.reshape(-1, 1),
            Y_notrct_rural_control.reshape(-1, 1),
            Y_urban_control.reshape(-1, 1),
        )
    ).ravel()

    # checked
    T_obs = np.array(
        [1] * int(Y_obs_treated_rural.shape[0] + Y_obs_treated_urban.shape[0])
        + [0] * int(Y_notrct_rural_control.shape[0] + Y_urban_control.shape[0])
    )

    assert Y_rct_eval.shape == T_rct_eval.shape
    assert Y_rct.shape == T_rct.shape
    assert Y_obs.shape == Y_obs.shape
    assert X_rct.shape[0] == T_rct.shape[0]
    assert X_obs.shape[0] == T_obs.shape[0]
    assert X_rct_eval.shape[0] == T_rct_eval.shape[0]

    return X_rct, Y_rct, T_rct, X_obs, Y_obs, T_obs, X_rct_eval, Y_rct_eval, T_rct_eval


def ite_adjusted_outcome(Y, T, _propensity, c):
    assert _propensity > 0
    assert _propensity < 1
    assert len(Y.shape) == 1
    assert Y.shape == T.shape
    _ite = T / _propensity * (Y - c) - (1 - T) / (1 - _propensity) * (Y - c)
    assert _ite.shape == Y.shape
    return _ite


class STARDataset:

    def __init__(
        self,
        data_path: str,
        cat_covar_columns=[
            "g1surban", #confounder, will be dropped in experiment
            "gender",
            "race",
            "birthmonth",
            "birthday",
            "birthyear",
            "gkfreelunch",
            "g1tchid",
            "g1freelunch",
        ],
    ):

        self.data_path = data_path

        # pre-process data
        star_data = pd.read_csv(self.data_path)

        (
            self.X_rural,
            self.X_urban,
            self.T_rural,
            self.T_urban,
            self.Y_rural,
            self.Y_urban,
            self._propensity_score,
            self._mean_outcome,
        ) = preprocess_star_dataset(star_data, cat_covar_columns)

    def get_propensity_score(self):
        return self._propensity_score

    def sample(
        self,
        n1: int,
        n0: int,
        rct_fraction_of_rural: float = 0.5,
        eval_fraction_of_rct: float = 0.5,
    ):

        X_rct, Y_rct, T_rct, X_obs, Y_obs, T_obs, X_eval, Y_eval, T_eval = (
            _generate_rct_obs(
                self.X_rural,
                self.Y_rural,
                self.T_rural,
                self.X_urban,
                self.Y_urban,
                self.T_urban,
                rct_fraction_of_rural=rct_fraction_of_rural,
                eval_fraction_of_rct=eval_fraction_of_rct,
            )
        )

        # subsample randomly n1 rows from X_rct, Y_rct, T_rct (without replacement)
        rct_indices = np.random.choice(X_rct.shape[0], n1, replace=False)
        X_rct = X_rct[rct_indices]
        Y_rct = Y_rct[rct_indices]
        T_rct = T_rct[rct_indices]

        # subsample randomly n0 rows from X_obs, Y_obs, T_obs (without replacement)
        obs_indices = np.random.choice(X_obs.shape[0], n0, replace=False)
        X_obs = X_obs[obs_indices]
        Y_obs = Y_obs[obs_indices]
        T_obs = T_obs[obs_indices]

        X_train = np.row_stack([X_rct, X_obs])
        S_train = np.row_stack(
            [np.ones((T_rct.shape[0], 1)), np.zeros((T_obs.shape[0], 1))]
        )
        A_train = np.row_stack([T_rct.reshape(-1, 1), T_obs.reshape(-1, 1)])
        Y_train = np.row_stack([Y_rct.reshape(-1, 1), Y_obs.reshape(-1, 1)])

        gt_adjusted_ite_eval = ite_adjusted_outcome(
            Y_eval,
            T_eval,
            _propensity=self._propensity_score,
            c=self._mean_outcome,
        )

        return X_train, S_train, A_train, Y_train, X_eval, gt_adjusted_ite_eval
