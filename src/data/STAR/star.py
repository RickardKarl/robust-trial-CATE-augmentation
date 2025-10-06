import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
Based on processing of STAR dataset by Kallus et al. (2018)
"""


def _to_categorical(X):
    assert type(X) == type(np.ones((1,)))
    return pd.get_dummies(pd.DataFrame(X.astype(str)), dummy_na=True).values.astype(int)


def preprocess_star_dataset(
    star_data: pd.DataFrame, cat_covar_columns: list, target_label: str
):

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
    if target_label == "rural":
        _mean_outcome = np.mean(Y_rural)
    elif target_label == "urban":
        _mean_outcome = np.mean(Y_urban)
    else:
        raise ValueError(f"Invalid target_label: {target_label}")

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


def _generate_target_ext(
    X_rural,
    Y_rural,
    T_rural,
    X_urban,
    Y_urban,
    T_urban,
    target_fraction_of_main_loc=0.5,
    eval_fraction_of_target=0.4,
):

    X_urban_control = X_urban[T_urban == 0]
    X_urban_treated = X_urban[T_urban == 1]

    Y_urban_control = Y_urban[T_urban == 0]
    Y_urban_treated = Y_urban[T_urban == 1]

    if 0.0 < target_fraction_of_main_loc < 1.0:

        (
            X_target,
            X_rural_not_target,
            Y_target,
            Y_rural_not_target,
            T_target,
            T_rural_not_target,
        ) = train_test_split(
            X_rural, Y_rural, T_rural, test_size=1 - target_fraction_of_main_loc
        )

    elif target_fraction_of_main_loc == 1.0:
        X_target = X_rural
        Y_target = Y_rural
        T_target = T_rural
        # empty arrays with the correct shapes
        X_rural_not_target = np.empty((0, X_rural.shape[1]))
        Y_rural_not_target = np.empty((0,))
        T_rural_not_target = np.empty((0,))
    else:
        raise ValueError(
            f"Invalid value for target_fraction_of_main_loc, but be in range (0,1] but got {target_fraction_of_main_loc}"
        )

    (
        X_target,
        X_target_eval,
        Y_target,
        Y_target_eval,
        T_target,
        T_target_eval,
    ) = train_test_split(
        X_target,
        Y_target,
        T_target,
        test_size=eval_fraction_of_target,
    )

    X_nottarget = np.vstack((X_rural_not_target, X_urban))
    Y_nottarget = np.hstack((Y_rural_not_target, Y_urban))
    T_nottarget = np.hstack((T_rural_not_target, T_urban))
    _local_rural_filter = np.array(
        [True] * X_rural_not_target.shape[0] + [False] * X_urban.shape[0]
    ).ravel()
    assert _local_rural_filter.shape == Y_nottarget.shape

    X_nottarget_rural = X_nottarget[_local_rural_filter]
    Y_nottarget_rural = Y_nottarget[_local_rural_filter]
    T_nottarget_rural = T_nottarget[_local_rural_filter]

    # checked
    X_nottarget_rural_treated = X_nottarget_rural[T_nottarget_rural == 1]
    X_nottarget_rural_control = X_nottarget_rural[T_nottarget_rural == 0]

    # checked
    Y_nottarget_rural_treated = Y_nottarget_rural[T_nottarget_rural == 1]
    Y_nottarget_rural_control = Y_nottarget_rural[T_nottarget_rural == 0]

    # OUTCOME BASED REMOVAL
    # _treated_rural_filter = Y_nottarget_rural_treated < np.median(Y_nottarget_rural_treated)
    # _treated_urban_filter = Y_urban_treated < np.median(Y_urban[T_urban == 1])
    _treated_rural_filter = np.ones(len(Y_nottarget_rural_treated), dtype=np.bool)
    _treated_urban_filter = np.ones(len(Y_urban_treated), dtype=np.bool)

    # remove large samples.
    X_ext_treated_rural = X_nottarget_rural_treated[_treated_rural_filter, :]
    Y_ext_treated_rural = Y_nottarget_rural_treated[_treated_rural_filter]

    X_ext_treated_urban = X_urban_treated[_treated_urban_filter, :]
    Y_ext_treated_urban = Y_urban_treated[_treated_urban_filter]
    assert X_ext_treated_rural.shape[0] == Y_ext_treated_rural.shape[0]

    # checked
    X_ext = np.vstack(
        (
            X_ext_treated_rural,
            X_ext_treated_urban,
            X_nottarget_rural_control,
            X_urban_control,
        )
    )

    # checked
    Y_ext = np.vstack(
        (
            Y_ext_treated_rural.reshape(-1, 1),
            Y_ext_treated_urban.reshape(-1, 1),
            Y_nottarget_rural_control.reshape(-1, 1),
            Y_urban_control.reshape(-1, 1),
        )
    ).ravel()

    # checked
    T_ext = np.array(
        [1] * int(Y_ext_treated_rural.shape[0] + Y_ext_treated_urban.shape[0])
        + [0] * int(Y_nottarget_rural_control.shape[0] + Y_urban_control.shape[0])
    )

    assert Y_target_eval.shape == T_target_eval.shape
    assert Y_target.shape == T_target.shape
    assert Y_ext.shape == Y_ext.shape
    assert X_target.shape[0] == T_target.shape[0]
    assert X_ext.shape[0] == T_ext.shape[0]
    assert X_target_eval.shape[0] == T_target_eval.shape[0]

    return (
        X_target,
        Y_target,
        T_target,
        X_ext,
        Y_ext,
        T_ext,
        X_target_eval,
        Y_target_eval,
        T_target_eval,
    )


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
            "g1surban",  # confounder, will be dropped in experiment
            "gender",
            "race",
            "birthmonth",
            "birthday",
            "birthyear",
            "gkfreelunch",
            "g1tchid",
            "g1freelunch",
        ],
        target_label="rural",
    ):

        self.data_path = data_path
        assert target_label in ["rural", "urban"]
        self.target_label = target_label

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
        ) = preprocess_star_dataset(star_data, cat_covar_columns, target_label)

    def get_propensity_score(self):
        return self._propensity_score

    def sample(
        self,
        n1: int,
        n0: int,
        target_fraction_of_main_loc: float = 0.5,
        eval_fraction_of_target: float = 0.5,
    ):

        if self.target_label == "rural":
            (
                X_target,
                Y_target,
                T_target,
                X_ext,
                Y_ext,
                T_ext,
                X_eval,
                Y_eval,
                T_eval,
            ) = _generate_target_ext(
                self.X_rural,
                self.Y_rural,
                self.T_rural,
                self.X_urban,
                self.Y_urban,
                self.T_urban,
                target_fraction_of_main_loc=target_fraction_of_main_loc,
                eval_fraction_of_target=eval_fraction_of_target,
            )
        else:
            (
                X_target,
                Y_target,
                T_target,
                X_ext,
                Y_ext,
                T_ext,
                X_eval,
                Y_eval,
                T_eval,
            ) = _generate_target_ext(
                self.X_urban,
                self.Y_urban,
                self.T_urban,
                self.X_rural,
                self.Y_rural,
                self.T_rural,
                target_fraction_of_main_loc=target_fraction_of_main_loc,
                eval_fraction_of_target=eval_fraction_of_target,
            )

        # subsample randomly n1 rows from X_target, Y_target, T_target (without replacement)
        target_indices = np.random.choice(X_target.shape[0], n1, replace=False)
        X_target = X_target[target_indices]
        Y_target = Y_target[target_indices]
        T_target = T_target[target_indices]

        # subsample randomly n0 rows from X_ext, Y_ext, T_ext (without replacement)
        ext_indices = np.random.choice(X_ext.shape[0], n0, replace=False)
        X_ext = X_ext[ext_indices]
        Y_ext = Y_ext[ext_indices]
        T_ext = T_ext[ext_indices]

        X_train = np.row_stack([X_target, X_ext])
        S_train = np.row_stack(
            [np.ones((T_target.shape[0], 1)), np.zeros((T_ext.shape[0], 1))]
        )
        A_train = np.row_stack([T_target.reshape(-1, 1), T_ext.reshape(-1, 1)])
        Y_train = np.row_stack([Y_target.reshape(-1, 1), Y_ext.reshape(-1, 1)])

        gt_adjusted_ite_eval = ite_adjusted_outcome(
            Y_eval,
            T_eval,
            _propensity=self._propensity_score,
            c=self._mean_outcome,
        )

        return X_train, S_train, A_train, Y_train, X_eval, gt_adjusted_ite_eval
