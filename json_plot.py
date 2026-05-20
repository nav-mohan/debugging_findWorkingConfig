#!/usr/bin/env python3

from pathlib import Path
import json
import math

import numpy as np
import matplotlib.pyplot as plt


PLOTDATA_DIR = Path("plotdata")
OUTPUT_DIR = Path("plots")


def unwrap_file_root(data, path):
    """
    Your files have structure:

        [
            {
                "avg_equil_alat": ...,
                "avg_good_alat": ...,
                "results": [...]
            }
        ]

    This function also tolerates the case where the root is already that dict.
    """
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(
                f"{path.name}: expected root list of length 1, got length {len(data)}"
            )

        if not isinstance(data[0], dict):
            raise ValueError(f"{path.name}: root list item is not a dictionary")

        return data[0]

    if isinstance(data, dict):
        return data

    raise ValueError(f"{path.name}: JSON root must be a list or dictionary")


def load_one_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    root = unwrap_file_root(data, path)

    required_file_keys = ["avg_equil_alat", "avg_good_alat", "results"]
    missing = [key for key in required_file_keys if key not in root]
    if missing:
        raise KeyError(f"{path.name}: missing file-level keys {missing}")

    if not isinstance(root["results"], list):
        raise TypeError(f"{path.name}: expected 'results' to be a list")

    return {
        "path": path,
        "avg_equil_alat": float(root["avg_equil_alat"]),
        "avg_good_alat": float(root["avg_good_alat"]),
        "results": root["results"],
    }


def load_all_files(folder):
    json_files = sorted(folder.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {folder.resolve()}")

    file_records = []
    for path in json_files:
        file_records.append(load_one_file(path))

    return file_records


def validate_spec(spec, path):
    required_spec_keys = [
        "spec",
        "spec_equil_alat",
        "spec_good_alat",
        "spec_valid_alats",
        "spec_valid_energy",
        "spec_min_indices",
        "spec_min_index",
    ]

    missing = [key for key in required_spec_keys if key not in spec]
    if missing:
        raise KeyError(f"{path.name}, spec={spec.get('spec', '<unknown>')}: missing {missing}")

    if len(spec["spec_valid_alats"]) != len(spec["spec_valid_energy"]):
        raise ValueError(
            f"{path.name}, spec={spec['spec']}: "
            f"len(spec_valid_alats)={len(spec['spec_valid_alats'])} but "
            f"len(spec_valid_energy)={len(spec['spec_valid_energy'])}"
        )


def choose_subplot_grid(n):
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def safe_indices(indices, n):
    good = []

    for idx in indices:
        if isinstance(idx, int) and 0 <= idx < n:
            good.append(idx)
        else:
            print(f"Warning: skipping invalid index {idx}; valid range is [0, {n - 1}]")

    return good


def plot_file_record(file_record, output_dir):
    path = file_record["path"]
    specs = file_record["results"]

    if not specs:
        print(f"Skipping {path.name}: no specs in results")
        return

    for spec in specs:
        validate_spec(spec, path)

    nplots = len(specs)
    nrows, ncols = choose_subplot_grid(nplots)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for ax, spec in zip(axes_flat, specs):
        spec_name = spec["spec"]

        alats = np.asarray(spec["spec_valid_alats"], dtype=float)
        energies = np.asarray(spec["spec_valid_energy"], dtype=float)

        spec_equil_alat = float(spec["spec_equil_alat"])
        spec_good_alat = float(spec["spec_good_alat"])

        spec_min_indices = safe_indices(spec["spec_min_indices"], len(alats))
        spec_min_index = spec["spec_min_index"]

        # Main energy curve
        ax.plot(
            alats,
            energies,
            marker=".",
            linestyle="-",
            linewidth=1.0,
            markersize=3,
            label="valid energy",
        )

        # Green points: all local/good candidate minima
        if spec_min_indices:
            ax.scatter(
                alats[spec_min_indices],
                energies[spec_min_indices],
                color="green",
                s=70,
                zorder=4,
                label="spec_min_indices",
            )

        # Red point: selected minimum
        if isinstance(spec_min_index, int) and 0 <= spec_min_index < len(alats):
            ax.scatter(
                alats[spec_min_index],
                energies[spec_min_index],
                color="red",
                s=100,
                zorder=5,
                label="spec_min_index",
            )
        else:
            print(
                f"Warning: {path.name}, spec={spec_name}: "
                f"invalid spec_min_index={spec_min_index}"
            )

        # Vertical line at equilibrium alat
        if spec_equil_alat != -1:
            ax.axvline(
                spec_equil_alat,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="spec_equil_alat",
            )

            # Textbox: per-spec difference
            spec_diff = 100*(1 - spec_good_alat/spec_equil_alat)
            textbox = (
                f"error = {spec_diff:.2g}%\n"
                f"equil = {spec_equil_alat:.4g}\n"
                f"good  = {spec_good_alat:.4g}"
            )

            ax.text(
                0.03,
                0.97,
                textbox,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "edgecolor": "gray",
                    "alpha": 0.85,
                },
            )

        ax.set_title(str(spec_name))
        ax.set_xlabel("spec_valid_alats")
        ax.set_ylabel("spec_valid_energy")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes_flat[nplots:]:
        ax.set_visible(False)

    avg_equil_alat = file_record["avg_equil_alat"]
    avg_good_alat = file_record["avg_good_alat"]
    avg_diff = 100*(1 - avg_good_alat/avg_equil_alat)

    fig.suptitle(
        (
            f"{path.name}\n"
            f"avg_equil_alat - avg_good_alat = {avg_diff:.2g} "
            f"({avg_equil_alat:.4g} - {avg_good_alat:.4g})"
        ),
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_spec_energy.png"

    fig.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

    plt.close(fig)


def main():
    file_records = load_all_files(PLOTDATA_DIR)

    for file_record in file_records:
        plot_file_record(file_record, OUTPUT_DIR)


if __name__ == "__main__":
    main()