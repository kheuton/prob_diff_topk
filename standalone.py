import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


DEFAULT_PLOTS = [
    "cook_2021_2022",
    "ma_2020_2021",
    "asurv_2009_2010",
]

DEFAULT_VIOLIN_HEIGHTS = {
    "cook_2021_2022": 0.5,
    "ma_2020_2021": 5.0,
    "asurv_2009_2010": 1.0,
}


def plot_styles():
    colors = {
        (30, 0): "#1f77b4",
        (30, 1): "#ff7f0e",
        (0, 1): "#2ca02c",
        (10, 10): "#9467bd",
        (20, 20): "#c39bd3",
    }
    markers = {
        (30, 0): "o",
        (30, 1): "s",
        (0, 1): "^",
        (10, 10): "D",
        (20, 20): "D",
    }
    labels = {
        (30, 0): "BPR Only",
        (30, 1): "DAML",
        (0, 1): "NLL Only",
        (10, 10): "SPO+",
        (20, 20): "PG",
    }
    return colors, markers, labels


def load_payload(path: Path):
    with path.open("r") as f:
        return json.load(f)


def draw_plot(payload: dict, violin_height: float, save_path: Path, show: bool = False):
    rows = pd.DataFrame(payload["selected_rows"])
    cfg = payload.get("plot_kwargs", {})

    location = cfg.get("location", payload.get("plot_name", "plot"))
    title_string = cfg.get("title_string", f"{location}, Frozen Test Performance")
    legend_loc = cfg.get("legend_loc", "lower left")
    x_trans = cfg.get("x_trans", 0.13)
    y_trans = cfg.get("y_trans", 0.13)
    break_point = cfg.get("break_point", None)
    upper_ylim = cfg.get("upper_ylim", None)
    lower_ylim = cfg.get("lower_ylim", None)

    colors, markers, labels = plot_styles()
    thresholds = sorted([float(t) for t in rows["tr"].dropna().unique()])
    shades_of_orange = list(
        mcolors.LinearSegmentedColormap.from_list("", ["#fd8d3c", "#f03b20"])(
            np.linspace(0, 1, max(len(thresholds), 1))
        )
    ) * 2

    if break_point is not None:
        fig, (ax, ax2) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(6, 5),
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax2 = None

    all_handles, all_labels = [], []

    for _, row in rows.iterrows():
        bw, nw = int(row["bw"]), int(row["nw"])
        tr = row["tr"] if pd.notna(row["tr"]) else None

        trial_vals = np.array(row["trial_test_bprs"], dtype=float)
        kde = gaussian_kde(trial_vals)
        x_range = np.linspace(trial_vals.min(), trial_vals.max(), 200)
        density = kde(x_range)
        density = density / density.max() * violin_height

        y_value = -float(row["test_nll"])
        y_lower = y_value - violin_height / 2
        y_upper = y_lower + density

        if tr is not None and len(thresholds) > 0:
            t_idx = thresholds.index(float(tr))
            color = shades_of_orange[(-t_idx + 5)] if len(shades_of_orange) > 5 else shades_of_orange[t_idx]
        else:
            color = colors[(bw, nw)]

        ax_to_use = ax2 if (break_point is not None and y_value < break_point) else ax
        ax_to_use.fill_between(x_range, y_lower, y_upper, color=color, alpha=0.5, zorder=0)
        sc = ax_to_use.scatter(
            float(row["avg_test_bpr"]),
            y_value,
            color=color,
            marker=markers[(bw, nw)],
            label=labels[(bw, nw)],
            zorder=1,
        )
        all_handles.append(sc)
        all_labels.append(sc.get_label())

    for method in [(10, 10), (20, 20)]:
        if labels[method] not in all_labels:
            dummy = ax.scatter([], [], color=colors[method], marker=markers[method], label=labels[method])
            all_handles.append(dummy)
            all_labels.append(labels[method])

    ax.set_title(title_string, fontsize=20)
    ax.set_ylabel("Test Log Likelihood", fontsize=18)
    (ax2 if ax2 is not None else ax).set_xlabel("Test BPR", fontsize=18)

    if break_point is not None:
        if upper_ylim is not None:
            ax.set_ylim(*upper_ylim)
        if lower_ylim is not None:
            ax2.set_ylim(*lower_ylim)
        ax.spines["bottom"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.xaxis.tick_top()
        ax2.xaxis.tick_bottom()
        d = 0.015
        kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        dy = 0.03
        ax2.plot((-d, +d), (1 - dy, 1 + dy), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - dy, 1 + dy), **kwargs)

    by_label = dict(sorted(zip(all_labels, all_handles), key=lambda x: x[0]))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=17,
        loc=legend_loc,
        bbox_to_anchor=(x_trans, y_trans),
        frameon=True,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Regenerate all frozen KDE scatter plots.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("frozen_plot_data"),
        help="Directory containing frozen JSON payloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frozen_plot_data"),
        help="Directory to write graph PDFs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively while generating files.",
    )
    args = parser.parse_args()

    for plot_name in DEFAULT_PLOTS:
        json_path = args.input_dir / f"{plot_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing frozen payload: {json_path}")

        payload = load_payload(json_path)
        violin_height = DEFAULT_VIOLIN_HEIGHTS.get(plot_name, 1.0)
        out_pdf = args.output_dir / f"{plot_name}.pdf"

        draw_plot(payload, violin_height=violin_height, save_path=out_pdf, show=args.show)
        print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
