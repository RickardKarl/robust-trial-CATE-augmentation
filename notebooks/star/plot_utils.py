import matplotlib.pyplot as plt


def set_default_matplotlib_settings():
    # Make figure font into times new roman
    plt.rcParams["font.family"] = "Times New Roman"


method_style_map = {
    "DR-learner": {"color": "green", "marker": "s", "linestyle": "--"},
    "T-learner": {"color": "red", "marker": "D", "linestyle": "--"},
    "Pooled T-learner": {"color": "purple", "marker": "^", "linestyle": ":"},
    "CFACE (Asaiee et al., 2023)": {"color": "orange", "marker": "v", "linestyle": "-"},
    "R-OSCAR (Asaiee et al., 2025)": {
        "color": "silver",
        "marker": ">",
        "linestyle": ":",
    },
    "Kallus et al. (2018)": {"color": "brown", "marker": "x", "linestyle": ":"},
    "QR-learner (ours)": {"color": "cyan", "marker": "*", "linestyle": "-"},
    "Combined learner (ours)": {
        "color": "#0072B2",
        "marker": "h",
        "linestyle": "--",
    },
}
