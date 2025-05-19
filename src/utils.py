import numpy as np
import pandas as pd


def compute_qini_curve(evaluation_data, score, n_bins=10) -> np.array:

    assert "A" in evaluation_data.columns and "Y" in evaluation_data.columns
    assert len(score) == len(evaluation_data)

    evaluation_data = evaluation_data.copy()

    evaluation_data["score"] = score
    evaluation_data = evaluation_data.sort_values(by="score", ascending=False)

    qini_curve = np.zeros(n_bins + 1)
    bin_size = len(evaluation_data) // n_bins

    n_treated = len(evaluation_data[evaluation_data["A"] == 1])
    n_control = len(evaluation_data[evaluation_data["A"] == 0])

    for i in range(1, n_bins + 1):
        bin_data = evaluation_data.iloc[: i * bin_size]
        treated = bin_data[bin_data["A"] == 1]
        control = bin_data[bin_data["A"] == 0]

        treated_outcome = treated["Y"].sum() / n_treated
        control_outcome = control["Y"].sum() / n_control

        qini_curve[i] = treated_outcome - control_outcome

    return qini_curve


def compute_area_under_qini_curve(
    evaluation_data, score, n_bins=10, normalize=True
) -> float:
    qini_curve = compute_qini_curve(evaluation_data, score, n_bins)
    area_under_qini_curve = np.trapz(qini_curve, dx=1.0 / n_bins)

    if normalize:
        linear_curve = np.linspace(0, qini_curve[-1], n_bins + 1)
        area_under_linear_curve = np.trapz(linear_curve, dx=1.0 / n_bins)
        area_under_qini_curve -= area_under_linear_curve

    return area_under_qini_curve
