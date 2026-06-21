#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected the JSON root to be a dictionary")

    return data


def result_panels(data: dict) -> list[dict]:
    """
    Return mono-species results followed by the mixed-species result.

    The JSON also contains a duplicated `results` list and `final_result`, so
    this function deliberately uses only `mono_species_results` and
    `mixed_species_result` to avoid plotting the same result twice.
    """
    panels = list(data.get("mono_species_results", []))

    mixed = data.get("mixed_species_result")
    if isinstance(mixed, dict):
        panels.append(mixed)

    return panels


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(value):
        return default

    return value


def finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError(f"x and y have different lengths: {len(x)} != {len(y)}")

    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def plot_one_result(ax, result: dict, approx_mixed_alat=None):
    label = result.get("species_label", "unknown")
    configuration_type = result.get("configuration_type", "unknown")

    coarse = result.get("coarse_scan", {})
    coarse_x, coarse_y = finite_xy(
        coarse.get("alats", []),
        coarse.get("energies_per_atom", []),
    )

    # The coarse scan is ordered in lattice constant, so a connected curve is
    # physically meaningful.
    if len(coarse_x):
        order = np.argsort(coarse_x)
        ax.plot(
            coarse_x[order],
            coarse_y[order],
            marker=".",
            markersize=4,
            linewidth=1.1,
            label="coarse lattice sweep",
        )

    # Coarse minima that became Nelder-Mead starting points.
    coarse_minima = result.get("coarse_minima", [])
    if coarse_minima:
        min_x = []
        min_y = []

        for row in coarse_minima:
            a = safe_float(row.get("alat"))
            e = safe_float(row.get("energy_per_atom"))
            if a is not None and e is not None:
                min_x.append(a)
                min_y.append(e)

        if min_x:
            ax.scatter(
                min_x,
                min_y,
                marker="v",
                s=55,
                zorder=5,
                label="coarse minima / NM starts",
            )

    # Nelder-Mead evaluations are shown as scatter points only. Connecting
    # these points would imply a sampled E(a) curve, but they are actually
    # trial evaluations in optimizer order.
    attempts = result.get("nelder_mead_attempts", [])
    for attempt_index, attempt in enumerate(attempts):
        evaluations = attempt.get("evaluations", [])
        nm_x = []
        nm_y = []

        for row in evaluations:
            a = safe_float(row.get("alat"))
            e = safe_float(row.get("energy_per_atom"))
            if a is not None and e is not None:
                nm_x.append(a)
                nm_y.append(e)

        if nm_x:
            attempt_label = (
                "selected NM evaluations"
                if attempt_index == result.get("selected_attempt_index")
                else f"NM attempt {attempt_index}"
            )
            ax.scatter(
                nm_x,
                nm_y,
                marker="o",
                s=22,
                alpha=0.65,
                zorder=4,
                label=attempt_label,
            )

    # Final selected equilibrium point.
    good_alat = safe_float(result.get("good_alat"))
    good_energy = safe_float(result.get("good_energy_per_atom"))

    if good_alat is not None and good_energy is not None:
        ax.scatter(
            [good_alat],
            [good_energy],
            marker="*",
            s=150,
            zorder=7,
            label="selected equilibrium",
        )
        ax.axvline(
            good_alat,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )

    # For the mixed-species panel, show the average mono-species estimate that
    # was used to define the mixed sweep.
    if (
        configuration_type == "mixed_species_fcc"
        and approx_mixed_alat is not None
    ):
        ax.axvline(
            approx_mixed_alat,
            linestyle=":",
            linewidth=1.3,
            label="average mono-species guess",
        )

    switched = coarse.get("switched_to_direct")
    switch_alat = safe_float(coarse.get("switch_alat"))
    plateau_alat = safe_float(coarse.get("plateau_stop_alat"))

    text_lines = [
        f"type: {configuration_type}",
        f"status: {result.get('status', 'unknown')}",
    ]

    if good_alat is not None:
        text_lines.append(f"equilibrium a: {good_alat:.6g} Å")
    if good_energy is not None:
        text_lines.append(f"minimum E: {good_energy:.6g} eV/atom")
    if switched:
        if switch_alat is None:
            text_lines.append("safe → direct: yes")
        else:
            text_lines.append(f"safe → direct at: {switch_alat:.4g} Å")
    if plateau_alat is not None:
        text_lines.append(f"plateau stop: {plateau_alat:.4g} Å")

    ax.text(
        0.98,
        0.97,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.85,
        },
    )

    ax.set_title(label)
    ax.set_xlabel("Lattice constant (Å)")
    ax.set_ylabel("Energy per atom (eV/atom)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def choose_grid(nplots: int):
    ncols = min(3, max(1, math.ceil(math.sqrt(nplots))))
    nrows = math.ceil(nplots / ncols)
    return nrows, ncols


def plot_json(path: Path, output_dir: Path, show: bool, dpi: int):
    data = load_json(path)
    panels = result_panels(data)

    if not panels:
        raise ValueError(f"{path}: no mono-species or mixed-species results found")

    nrows, ncols = choose_grid(len(panels))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 5.2 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    approx_mixed_alat = safe_float(data.get("approx_mixed_equilibrium_alat"))

    for ax, result in zip(axes_flat, panels):
        plot_one_result(
            ax=ax,
            result=result,
            approx_mixed_alat=approx_mixed_alat,
        )

    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    model_name = data.get("model_shortname") or data.get("model") or path.stem
    species_label = data.get("species_label", "")
    final_alat = safe_float(data.get("equilibrium_alat"))

    title = (
        f"{model_name}\n"
        f"FCC equilibrium search: {species_label}; "
        f"ncells/side = {data.get('ncells_per_side', '?')}"
    )
    if final_alat is not None:
        title += f"; final mixed equilibrium = {final_alat:.6g} Å"

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_equilibrium_curves.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {output_path}")

    if show:
        try:
            fig.canvas.manager.set_window_title(path.stem)
        except Exception:
            pass
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot mono-species and mixed-species FCC equilibrium-search JSON files."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A JSON file or a directory containing JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots_equilibrium_config_fcc"),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    if args.input_path.is_file():
        json_paths = [args.input_path]
    elif args.input_path.is_dir():
        json_paths = sorted(args.input_path.glob("*.json"))
    else:
        raise FileNotFoundError(args.input_path)

    if not json_paths:
        raise FileNotFoundError(
            f"No JSON files found under {args.input_path.resolve()}"
        )

    failures = []
    for path in json_paths:
        try:
            plot_json(
                path=path,
                output_dir=args.output_dir,
                show=args.show,
                dpi=args.dpi,
            )
        except Exception as exc:
            failures.append((path, exc))
            print(f"FAILED {path}: {exc}")

    if failures:
        raise RuntimeError(
            f"Failed to plot {len(failures)} of {len(json_paths)} JSON files"
        )


if __name__ == "__main__":
    main()
