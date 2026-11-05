import matplotlib.pyplot as plt
import numpy as np


def readmission_rate_by(data, col, target="readmitted", top_n=15, min_count=100, sort_by="rate", ax=None):
    """Plot the 30-day readmission rate per category with 95% CI and sample sizes."""
    g = data.groupby(col)[target].agg(rate="mean", n="size").query("n >= @min_count").sort_values("n", ascending=False).head(top_n)
    g.index = g.index.astype(str)
    g = g.sort_index() if sort_by == "index" else g.sort_values(sort_by)

    z = 1.96
    p, n = g["rate"], g["n"]
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    lower_err = (p - (centre - half)).clip(lower=0)
    upper_err = ((centre + half) - p).clip(lower=0)

    ax = ax or plt.subplots(figsize=(7, 0.4 * len(g) + 1.5))[1]
    ax.barh(g.index, g["rate"], xerr=[lower_err, upper_err], color="slateblue", alpha=0.85)
    ax.axvline(data[target].mean(), color="red", ls="--", label="overall rate")
    xmax = (g["rate"] + upper_err).max()
    for i, (r, n) in enumerate(zip(g["rate"], g["n"])):
        ax.text(xmax * 1.02, i, f"n={n:,}", va="center", fontsize=8, color="black")
    ax.set_xlim(0, xmax * 1.2)
    ax.set_xlabel("30-day readmission rate")
    ax.set_title(f"Readmission rate by {col}")
    ax.legend(loc="lower right")
    return ax
