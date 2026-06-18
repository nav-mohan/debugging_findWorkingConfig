"""
python plot_fwc_nm_results.py \
  --json-dir plotdata_nm \
  --output-dir fwc_nm_plots
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def finite_xy(x, y):
    x = np.asarray(as_list(x), dtype=float)
    y = np.asarray(as_list(y), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([])
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def sort_xy(x, y):
    x, y = finite_xy(x, y)
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    return x[order], y[order]


def nearest_y(x, y, x0):
    x, y = finite_xy(x, y)
    if len(x) == 0 or x0 is None:
        return None
    idx = int(np.argmin(np.abs(x - float(x0))))
    return float(y[idx])


def sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_records(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Unexpected JSON root in {json_path}: {type(data)}")


def candidate_points_from_fallback(result):
    accepted_x = []
    accepted_y = []
    accepted_labels = []

    for cand in result.get("accepted_candidates", []):
        if "alat" in cand and "energy_per_atom" in cand:
            accepted_x.append(cand["alat"])
            accepted_y.append(cand["energy_per_atom"])
            accepted_labels.append(cand.get("curvature"))
            continue

        best = cand.get("best", {})
        if "alat" in best and "energy_per_atom" in best:
            accepted_x.append(best["alat"])
            accepted_y.append(best["energy_per_atom"])
            accepted_labels.append(best.get("curvature"))

    rejected_x = []
    rejected_y = []

    for cand in result.get("rejected_candidates", []):
        if "alat" in cand and "energy_per_atom" in cand:
            rejected_x.append(cand["alat"])
            rejected_y.append(cand["energy_per_atom"])
            continue

        wx = cand.get("window_alats", cand.get("level2_alats", []))
        wy = cand.get(
            "window_energies_per_atom",
            cand.get("level2_energies_per_atom", []),
        )
        x, y = finite_xy(wx, wy)
        if len(x) > 0:
            idx = int(np.argmin(y))
            rejected_x.append(float(x[idx]))
            rejected_y.append(float(y[idx]))

    return accepted_x, accepted_y, accepted_labels, rejected_x, rejected_y


def plot_species_axis(ax, result):
    species = result.get("species", "?")
    strategy = result.get("search_strategy", "?")

    coarse_x, coarse_y = sort_xy(
        result.get("coarse_scan_alats", []),
        result.get("coarse_scan_energies_per_atom", []),
    )

    if len(coarse_x) > 0:
        ax.plot(
            coarse_x,
            coarse_y,
            color="0.35",
            linewidth=1.0,
            marker=".",
            markersize=3,
            label="coarse scan",
        )

    for a in result.get("coarse_min_alats", []):
        y = nearest_y(coarse_x, coarse_y, a)
        if y is not None:
            ax.scatter(
                [a],
                [y],
                marker="v",
                s=45,
                color="tab:blue",
                edgecolor="black",
                linewidth=0.3,
                zorder=5,
                label="coarse minima",
            )

    trace_x, trace_y = finite_xy(
        result.get("all_alats", []),
        result.get("all_energies_per_atom", []),
    )

    if len(trace_x) > 0:
        if strategy == "nelder_mead":
            ax.plot(
                trace_x,
                trace_y,
                color="tab:orange",
                linewidth=0.6,
                alpha=0.35,
                label="NM trace order",
            )
            ax.scatter(
                trace_x,
                trace_y,
                s=18,
                color="tab:orange",
                alpha=0.75,
                label="NM evaluations",
            )
        else:
            sx, sy = sort_xy(trace_x, trace_y)
            ax.scatter(
                sx,
                sy,
                s=14,
                color="tab:purple",
                alpha=0.65,
                label="fallback scans",
            )

    for a in result.get("level1_min_alats", []):
        ax.axvline(
            float(a),
            color="tab:purple",
            linewidth=0.7,
            linestyle=":",
            alpha=0.5,
        )

    acc_x, acc_y, acc_curv, rej_x, rej_y = candidate_points_from_fallback(result)
    if len(acc_x) > 0:
        ax.scatter(
            acc_x,
            acc_y,
            marker="D",
            s=46,
            color="tab:green",
            edgecolor="black",
            linewidth=0.4,
            zorder=7,
            label="accepted fallback",
        )

    if len(rej_x) > 0:
        ax.scatter(
            rej_x,
            rej_y,
            marker="x",
            s=45,
            color="tab:red",
            linewidth=1.1,
            zorder=6,
            label="rejected fallback",
        )

    good_a = result.get("good_alat", -1.0)
    good_e = result.get("good_energy_per_atom")
    if good_e is None:
        good_e = nearest_y(trace_x, trace_y, good_a)
    if good_a is not None and good_a > 0 and good_e is not None:
        ax.scatter(
            [good_a],
            [good_e],
            marker="*",
            s=150,
            color="crimson",
            edgecolor="black",
            linewidth=0.5,
            zorder=10,
            label="selected",
        )

    equil_a = result.get("spec_equil_alat", -1.0)
    if equil_a is not None and equil_a > 0:
        ax.axvline(
            float(equil_a),
            color="tab:green",
            linewidth=1.2,
            linestyle="--",
            label="equil alat",
        )

    rel = result.get("relative_difference_percent")
    curvature = result.get("good_curvature")
    subtitle = [
        f"{species}",
        f"strategy: {strategy}",
        f"good a={good_a:.5g}" if good_a is not None else "good a=?",
        f"good e={good_e:.5g}" if good_e is not None else "good e=?",
    ]
    if equil_a is not None and equil_a > 0:
        subtitle.append(f"equil a={equil_a:.5g}")
    if rel is not None:
        subtitle.append(f"reld={rel:.2f}%")
    if curvature is not None:
        subtitle.append(f"curv={curvature:.3g}")

    ax.set_title(" | ".join(subtitle), fontsize=9)
    ax.set_xlabel("lattice constant, a (A)")
    ax.set_ylabel("energy / atom")
    ax.grid(True, alpha=0.25)


def unique_legend(fig, axes):
    handles = []
    labels = []
    seen = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)

    if len(handles) > 0:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(5, len(handles)),
            fontsize=8,
            frameon=False,
        )


def plot_model_record(record, json_path, output_dir, write_pdf_page=None):
    results = record.get("results", [])
    if len(results) == 0:
        return None

    model_shortname = record.get("model_shortname", json_path.stem)
    model = record.get("model", json_path.stem)

    n = len(results)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.2 * ncols, 4.3 * nrows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for ax, result in zip(flat_axes, results):
        plot_species_axis(ax, result)

    for ax in flat_axes[len(results):]:
        ax.axis("off")

    avg_good = record.get("avg_good_alat")
    avg_equil = record.get("avg_equil_alat")
    title = f"{model_shortname}\n{model}"
    if avg_good is not None:
        title += f"\navg good a={avg_good:.5g}"
    if avg_equil is not None and avg_equil > 0:
        title += f" | avg equil a={avg_equil:.5g}"
    
    fig.suptitle(title, fontsize=12)
    unique_legend(fig, flat_axes[: len(results)])
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.92))
    plt.gcf().canvas.manager.set_window_title(f"{json_path.stem}_fwc_nm")

    mpl.rcParams['savefig.directory'] = output_dir

    plt.show()
    
    png_path = output_dir / f"{sanitize_filename(json_path.stem)}.png"
    # fig.savefig(png_path, dpi=180)

    # if write_pdf_page is not None:
        # write_pdf_page.savefig(fig)

    # plt.close(fig)
    return png_path


def write_summary_csv(rows, output_dir):
    path = output_dir / "summary.csv"
    columns = [
        "json_file",
        "model_shortname",
        "model",
        "species",
        "search_strategy",
        "good_alat",
        "good_energy_per_atom",
        "spec_equil_alat",
        "relative_difference_percent",
        "good_curvature",
        "num_nm_attempts",
        "num_coarse_minima",
        "num_accepted_candidates",
        "num_rejected_candidates",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})

    return path


def collect_summary_rows(json_path, record):
    rows = []
    for result in record.get("results", []):
        rows.append(
            {
                "json_file": json_path.name,
                "model_shortname": record.get("model_shortname"),
                "model": record.get("model"),
                "species": result.get("species"),
                "search_strategy": result.get("search_strategy"),
                "good_alat": result.get("good_alat"),
                "good_energy_per_atom": result.get("good_energy_per_atom"),
                "spec_equil_alat": result.get("spec_equil_alat"),
                "relative_difference_percent": result.get(
                    "relative_difference_percent"
                ),
                "good_curvature": result.get("good_curvature"),
                "num_nm_attempts": len(result.get("nelder_mead_attempts", [])),
                "num_coarse_minima": len(result.get("coarse_min_alats", [])),
                "num_accepted_candidates": len(
                    result.get("accepted_candidates", [])
                ),
                "num_rejected_candidates": len(
                    result.get("rejected_candidates", [])
                ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", required=True)
    parser.add_argument("--output-dir", default="fwc_nm_plots")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--pdf", default="fwc_nm_plots.pdf")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(json_dir.glob(args.pattern))
    if len(json_paths) == 0:
        raise FileNotFoundError(f"No JSON files matched {json_dir / args.pattern}")

    summary_rows = []
    png_paths = []
    pdf_path = output_dir / args.pdf

    with PdfPages(pdf_path) as pdf:
        for json_path in json_paths:
            records = load_records(json_path)
            for record in records:
                png_path = plot_model_record(
                    record=record,
                    json_path=json_path,
                    output_dir=output_dir,
                    write_pdf_page=pdf,
                )
                if png_path is not None:
                    png_paths.append(png_path)
                summary_rows.extend(collect_summary_rows(json_path, record))

    summary_path = write_summary_csv(summary_rows, output_dir)

    print(f"Wrote {len(png_paths)} PNG plot files to {output_dir}")
    print(f"Wrote combined PDF: {pdf_path}")
    print(f"Wrote summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
