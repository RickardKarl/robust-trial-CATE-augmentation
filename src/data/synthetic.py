import numpy as np
import pandas as pd


class Synthetic:

    def __init__(
        self, population_shift: float = 0.0, n_features: int = 5, seed: int = None
    ):

        self.population_shift = population_shift
        self.n_features = n_features
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def get_covar(self):
        return [f"X{i}" for i in range(self.n_features)]


    def control_outcome(self, x):
        terms = []
        for i in range(self.n_features):
            terms.append(3 / self.n_features * np.cos(3 / 2 * x[:, i]))
            for j in range(self.n_features):
                terms.append(1 / self.n_features * x[:, i] * x[:, j])
        return sum(terms)

    def modifier(self, x, s):
        terms = []
        for i in range(self.n_features):
            terms.append(x[:, i] / self.n_features)

        return sum(terms)

    def treatment_probability_external(self, x):
        linear_combination = np.sum(x, axis=1)
        intercept = -np.mean(linear_combination)
        return 1 / (1 + np.exp(-(intercept + linear_combination)))

    def sample(self, n_trial, n_obs) -> pd.DataFrame:

        n = n_trial + n_obs

        # Study assignment
        S_obs = np.zeros(n_obs, dtype=int)
        S_trial = np.ones(n_trial, dtype=int)
        S = np.concatenate((S_obs, S_trial))

        # Covariates
        sigma = np.full((self.n_features, self.n_features), 1 / 10)
        np.fill_diagonal(sigma, 1)
        X_obs = np.random.multivariate_normal(
            mean=np.zeros((self.n_features)) + self.population_shift,
            cov=sigma / np.sqrt(self.n_features),
            size=n_obs,
        )
        X_trial = np.random.multivariate_normal(
            mean=np.zeros((self.n_features)),
            cov=sigma / np.sqrt(self.n_features),
            size=n_trial,
        )
        X = np.vstack((X_obs, X_trial))

        # Treatment assignment in nonrandonmized experiment
        prob = self.treatment_probability_external(X[(S == 0).ravel(), :])
        A_S0 = np.random.binomial(1, prob)

        while np.all(A_S0 == 1) or np.all(A_S0 == 0):
            assert len(A_S0) > 1
            A_S0 = np.random.binomial(1, prob)

        # Treatment assignment in randomized experiment
        A0_trial = np.zeros(len(S_trial) // 2, dtype=int)
        A1_trial = np.ones(n_trial - len(A0_trial), dtype=int)
        A = np.concatenate((A_S0, A0_trial, A1_trial))

        Y0 = self.control_outcome(X)
        Y1 = Y0 + self.modifier(X, S)
        noise = np.random.normal(scale=1 / 2, size=n)
        Y = A * Y1 + (1 - A) * Y0 + noise

        return pd.DataFrame(
            {
                "A": A,
                "Y": Y,
                **{f"X{i}": X[:, i] for i in range(X.shape[1])},
                **{f"X{i}^2": X[:, i] ** 2 for i in range(X.shape[1])},
                "S": S,
                "Y1": Y1,
                "Y0": Y0,
                "cate": Y1 - Y0,
            }
        )


class SparseModifierSynthetic(Synthetic):

    def __init__(
        self,
        population_shift: float = 0,
        n_features: int = 5,
        effect_modifiers: list = [1 / 5],
        seed=None,
    ):
        super().__init__(population_shift, n_features, seed)
        assert len(effect_modifiers) <= n_features
        self.effect_modifiers = effect_modifiers

    def control_outcome(self, x):
        terms = []
        for i in range(self.n_features):
            terms.append(1 / self.n_features * x[:, i])
        return sum(terms)

    def modifier(self, x, s):
        terms = []
        for i, e in enumerate(self.effect_modifiers):
            terms.append((e + (1 - s) / 20) * x[:, i])

        return sum(terms)

