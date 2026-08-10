from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = Path("eval") / "eval_results"
OUTPUT_DIR = RESULT_DIR / "figures"
REFERENCE_DIR = (
    Path("outputs")
    / "mlp_diffusion"
    / "resolution_national_1k_lam0_mint30_200k"
)

MODELS = {
    "MLP": "qbin_mlp_national_n30_1k.csv",
    "cVAE": "qbin_cvae_national_n30_1k.csv",
    "cGAN": "qbin_cgan_national_n30_1k.csv",
    "MDN": "qbin_mdn_national_n30_1k.csv",
    "Conditional DDPM": "qbin_national_lam0_mint0_1k_seed200.csv",
    "Conditional DDPM + Snapping": "qbin_national_lam0_mint30_1k_seed200.csv",
}

STYLES = {
    "MLP": dict(color="#7f7f7f", marker="o", linestyle="--"),
    "cVAE": dict(color="#9467bd", marker="s", linestyle="--"),
    "cGAN": dict(color="#d62728", marker="^", linestyle="--"),
    "MDN": dict(color="#2ca02c", marker="D", linestyle="--"),
    "Conditional DDPM": dict(color="#1f77b4", marker="P", linestyle="-"),
    "Conditional DDPM + Snapping": dict(
        color="#ff7f0e", marker="X", linestyle="-"
    ),
}


def load_nrmse() -> dict[str, np.ndarray]:
    values = {}
    for model, filename in MODELS.items():
        frame = pd.read_csv(RESULT_DIR / filename)
        frame = frame[frame["bin"].astype(str).str.lower() != "all"].copy()
        frame["q"] = frame["bin"].str.extract(r"q(\d+)", expand=False).astype(int)
        frame = frame.sort_values("q")

        if frame["q"].tolist() != list(range(1, 11)):
            raise ValueError(f"{filename} does not contain exactly Q1--Q10.")
        values[model] = frame["NRMSE"].to_numpy(dtype=float)
    return values


def load_valid_counts() -> tuple[dict[str, np.ndarray], np.ndarray]:
    valid_counts = {}
    for model, filename in MODELS.items():
        frame = pd.read_csv(RESULT_DIR / filename)
        frame = frame[frame["bin"].astype(str).str.lower() != "all"].copy()
        frame["q"] = frame["bin"].str.extract(r"q(\d+)", expand=False).astype(int)
        frame = frame.sort_values("q")
        valid_counts[model] = frame["n"].to_numpy(dtype=int)

    target_counts = []
    for q in range(1, 11):
        matches = list(REFERENCE_DIR.glob(f"q{q}_*.csv"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one reference CSV for Q{q}, found {len(matches)}."
            )
        target_counts.append(len(pd.read_csv(matches[0])))

    return valid_counts, np.asarray(target_counts, dtype=int)


def draw(values: dict[str, np.ndarray], log_scale: bool) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(1, 11)

    for model, y in values.items():
        emphasis = model.startswith("Conditional DDPM")
        ax.plot(
            x,
            y,
            label=model,
            linewidth=2.2 if emphasis else 1.4,
            markersize=6 if emphasis else 4.5,
            markeredgewidth=0.8,
            **STYLES[model],
        )

    ax.set_xlabel("PDR quantile bin")
    ax.set_ylabel("NRMSE (lower is better)")
    ax.set_xticks(x, [f"Q{i}" for i in x])
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    suffix = "linear"
    if log_scale:
        ax.set_yscale("log")
        suffix = "log"

    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.4,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"qbin_nrmse_by_model_{suffix}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def draw_linear_with_zoom(values: dict[str, np.ndarray]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.5),
        sharex=True,
        gridspec_kw={"wspace": 0.18},
    )
    x = np.arange(1, 11)

    for ax in axes:
        for model, y in values.items():
            emphasis = model.startswith("Conditional DDPM")
            ax.plot(
                x,
                y,
                label=model,
                linewidth=2.1 if emphasis else 1.3,
                markersize=5.5 if emphasis else 4.2,
                markeredgewidth=0.8,
                **STYLES[model],
            )

        ax.set_xticks(x, [f"Q{i}" for i in x])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("PDR quantile bin")

    axes[0].set_ylim(0.0, 3.6)
    axes[0].set_ylabel("NRMSE (lower is better)")
    axes[0].set_title("(a) Full range")

    axes[1].set_ylim(0.0, 0.5)
    axes[1].set_title("(b) Zoomed range")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"qbin_nrmse_by_model_linear_zoom.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def draw_valid_counts(
    valid_counts: dict[str, np.ndarray], target_counts: np.ndarray
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(1, 11)
    width = 0.125
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * width

    for offset, (model, counts) in zip(offsets, valid_counts.items()):
        ax.bar(
            x + offset,
            counts,
            width=width,
            label=model,
            color=STYLES[model]["color"],
            edgecolor="none",
            alpha=0.9,
        )

    ax.plot(
        x,
        target_counts,
        color="#222222",
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        linestyle="--",
        label="Target samples",
        zorder=5,
    )

    ax.set_xlabel("PDR quantile bin")
    ax.set_ylabel(r"Number of valid samples ($n_{\mathrm{valid}}$)")
    ax.set_xticks(x, [f"Q{i}" for i in x])
    ax.set_ylim(0, max(target_counts) * 1.18)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.2,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"qbin_valid_samples_by_model.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    nrmse_values = load_nrmse()
    model_valid_counts, bin_target_counts = load_valid_counts()
    draw(nrmse_values, log_scale=False)
    draw(nrmse_values, log_scale=True)
    draw_linear_with_zoom(nrmse_values)
    draw_valid_counts(model_valid_counts, bin_target_counts)
    print(f"Saved figures to: {OUTPUT_DIR.resolve()}")
