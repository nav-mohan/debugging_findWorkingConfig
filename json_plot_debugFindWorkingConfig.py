#!/usr/bin/env python3

from pathlib import Path
import json
import math

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
PLOTDATA_DIR = Path("plotdata_multires")
OUTPUT_DIR = Path("plots_multires")

SAVE_FIGURES, SHOW_FIGURES = False,True
# SAVE_FIGURES, SHOW_FIGURES = True,False

# Plot controls for the new multi-resolution output
SHOW_ALL_SAMPLES = True
SHOW_VALID_SAMPLES = True
SHOW_LEVEL_MARKERS = True
SHOW_FINAL_WINDOWS = True
SHOW_ACCEPTED_CANDIDATES = True
SHOW_REJECTED_CANDIDATES = True
ANNOTATE_CURVATURE = True

# For huge level-2 scans, plotting every point can be slow/noisy.
# Set to None to plot all points.
MAX_POINTS_PER_CURVE = 5000


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def first_present(d, keys, default=None):
    """Return the first present non-None value from a dictionary."""
    for key in keys:
        if isinstance(d, dict) and key in d and d[key] is not None:
            return d[key]
    return default


def as_float_or_none(x):
    try:
        if x is None:
            return None
        val = float(x)
        if np.isfinite(val):
            return val
        return None
    except Exception:
        return None


def as_array(x, dtype=float):
    if x is None:
        return np.asarray([], dtype=dtype)
    try:
        return np.asarray(x, dtype=dtype)
    except Exception:
        return np.asarray([], dtype=dtype)


def downsample_xy(x, y, max_points=MAX_POINTS_PER_CURVE):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

    if max_points is None or len(x) <= max_points:
        return x, y

    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx], y[idx]


def choose_subplot_grid(n):
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def safe_indices(indices, n):
    good = []
    if indices is None:
        return good

    for idx in indices:
        try:
            idx = int(idx)
        except Exception:
            print(f"Warning: skipping non-integer index {idx}")
            continue

        if 0 <= idx < n:
            good.append(idx)
        else:
            print(f"Warning: skipping invalid index {idx}; valid range is [0, {n - 1}]")

    return good


def finite_difference_curvature(alats, energies, i):
    """
    Estimate local curvature d²E/da² at index i.

    energies should be energy-per-atom, not total energy.
    """
    alats = np.asarray(alats, dtype=float)
    energies = np.asarray(energies, dtype=float)

    if i is None:
        return None

    i = int(i)
    if i <= 0 or i >= len(energies) - 1:
        return None

    h_left = alats[i] - alats[i - 1]
    h_right = alats[i + 1] - alats[i]

    if not np.isfinite(h_left) or not np.isfinite(h_right):
        return None

    if abs(h_left) == 0 or abs(h_right) == 0:
        return None

    # Nonuniform-grid second derivative:
    # f''(x_i) ≈ 2 * [ h_l f_{i+1} - (h_l+h_r) f_i + h_r f_{i-1} ]
    #           / [ h_l h_r (h_l+h_r) ]
    return 2.0 * (
        h_left * energies[i + 1]
        - (h_left + h_right) * energies[i]
        + h_right * energies[i - 1]
    ) / (h_left * h_right * (h_left + h_right))


def nearest_index(x, value):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or value is None:
        return None
    return int(np.argmin(np.abs(x - value)))


# ---------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------
def unwrap_file_root(data, path):
    """
    Supports both old and new result files.

    Common structure:
        [
            {
                "avg_equil_alat": ...,
                "avg_good_alat": ...,
                "results": [...]
            }
        ]

    Also tolerates the case where the root is already that dict.
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

    if "results" not in root:
        raise KeyError(f"{path.name}: missing required key 'results'")

    if not isinstance(root["results"], list):
        raise TypeError(f"{path.name}: expected 'results' to be a list")

    return {
        "path": path,
        "avg_equil_alat": as_float_or_none(root.get("avg_equil_alat")),
        "avg_good_alat": as_float_or_none(root.get("avg_good_alat")),
        "results": root["results"],
        "root": root,
    }


def load_all_files(folder):
    json_files = sorted(folder.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {folder.resolve()}")

    return [load_one_file(path) for path in json_files]


# ---------------------------------------------------------------------
# Normalization layer: old output and new multi-resolution output
# ---------------------------------------------------------------------
def normalize_spec(spec, path):
    """
    Convert old and new result dictionaries into one plotting schema.

    Old per-spec keys include:
        spec_valid_alats
        spec_valid_energy_per_atom
        spec_min_indices
        spec_min_index
        spec_all_alats
        spec_all_energies_per_atom

    New multi-resolution keys may include:
        good_alat / spec_good_alat
        good_energy_per_atom
        good_curvature
        coarse_min_alats
        level1_min_alats
        accepted_candidates
        rejected_candidates
        search_strategy
        valid_alats / all_alats
    """
    name = first_present(spec, ["spec", "species", "symbol"], "<unknown>")

    equil_alat = as_float_or_none(first_present(spec, ["spec_equil_alat", "equil_alat"]))
    good_alat = as_float_or_none(first_present(spec, ["spec_good_alat", "good_alat"]))
    good_energy = as_float_or_none(first_present(spec, ["spec_good_energy_per_atom", "good_energy_per_atom"]))
    good_curvature = as_float_or_none(first_present(spec, ["spec_good_curvature", "good_curvature"]))

    all_alats = as_array(first_present(spec, ["spec_all_alats", "all_alats"]))
    all_energy = as_array(first_present(spec, ["spec_all_energies_per_atom", "all_energies_per_atom"]))

    valid_alats = as_array(first_present(spec, ["spec_valid_alats", "valid_alats"]))
    valid_energy = as_array(first_present(spec, ["spec_valid_energy_per_atom", "valid_energy_per_atom"]))

    min_indices = first_present(spec, ["spec_min_indices", "indices", "min_indices"], [])
    min_index = first_present(spec, ["spec_min_index", "min_index"], None)

    # New multi-resolution diagnostics
    coarse_min_alats = as_array(first_present(spec, ["spec_coarse_min_alats", "coarse_min_alats"]))
    level1_min_alats = as_array(first_present(spec, ["spec_level1_min_alats", "level1_min_alats"]))

    accepted_candidates = first_present(spec, ["spec_accepted_candidates", "accepted_candidates"], [])
    rejected_candidates = first_present(spec, ["spec_rejected_candidates", "rejected_candidates"], [])
    search_strategy = first_present(spec, ["spec_search_strategy", "search_strategy"], None)

    # Some wrappers may store the raw find_working_configuration dict under "result".
    nested = spec.get("result") if isinstance(spec, dict) else None
    if isinstance(nested, dict):
        if len(all_alats) == 0:
            all_alats = as_array(first_present(nested, ["all_alats", "spec_all_alats"]))
        if len(all_energy) == 0:
            all_energy = as_array(first_present(nested, ["all_energies_per_atom", "spec_all_energies_per_atom"]))
        if len(valid_alats) == 0:
            valid_alats = as_array(first_present(nested, ["valid_alats", "spec_valid_alats"]))
        if len(valid_energy) == 0:
            valid_energy = as_array(first_present(nested, ["valid_energy_per_atom", "spec_valid_energy_per_atom"]))
        if good_alat is None:
            good_alat = as_float_or_none(first_present(nested, ["good_alat", "spec_good_alat"]))
        if good_energy is None:
            good_energy = as_float_or_none(first_present(nested, ["good_energy_per_atom", "spec_good_energy_per_atom"]))
        if good_curvature is None:
            good_curvature = as_float_or_none(first_present(nested, ["good_curvature", "spec_good_curvature"]))
        if len(coarse_min_alats) == 0:
            coarse_min_alats = as_array(first_present(nested, ["coarse_min_alats", "spec_coarse_min_alats"]))
        if len(level1_min_alats) == 0:
            level1_min_alats = as_array(first_present(nested, ["level1_min_alats", "spec_level1_min_alats"]))
        if not accepted_candidates:
            accepted_candidates = first_present(nested, ["accepted_candidates", "spec_accepted_candidates"], [])
        if not rejected_candidates:
            rejected_candidates = first_present(nested, ["rejected_candidates", "spec_rejected_candidates"], [])
        if search_strategy is None:
            search_strategy = first_present(nested, ["search_strategy", "spec_search_strategy"], None)

    if len(all_alats) != len(all_energy):
        n = min(len(all_alats), len(all_energy))
        print(
            f"Warning: {path.name}, spec={name}: truncating all arrays "
            f"from ({len(all_alats)}, {len(all_energy)}) to {n}"
        )
        all_alats = all_alats[:n]
        all_energy = all_energy[:n]

    if len(valid_alats) != len(valid_energy):
        n = min(len(valid_alats), len(valid_energy))
        print(
            f"Warning: {path.name}, spec={name}: truncating valid arrays "
            f"from ({len(valid_alats)}, {len(valid_energy)}) to {n}"
        )
        valid_alats = valid_alats[:n]
        valid_energy = valid_energy[:n]

    return {
        "raw": spec,
        "name": name,
        "equil_alat": equil_alat,
        "good_alat": good_alat,
        "good_energy_per_atom": good_energy,
        "good_curvature": good_curvature,
        "all_alats": all_alats,
        "all_energy_per_atom": all_energy,
        "valid_alats": valid_alats,
        "valid_energy_per_atom": valid_energy,
        "min_indices": min_indices,
        "min_index": min_index,
        "coarse_min_alats": coarse_min_alats,
        "level1_min_alats": level1_min_alats,
        "accepted_candidates": accepted_candidates if isinstance(accepted_candidates, list) else [],
        "rejected_candidates": rejected_candidates if isinstance(rejected_candidates, list) else [],
        "search_strategy": search_strategy,
    }


# ---------------------------------------------------------------------
# Candidate plotting helpers
# ---------------------------------------------------------------------
def candidate_alat(candidate):
    return as_float_or_none(first_present(candidate, [
        "alat",
        "good_alat",
        "min_alat",
        "candidate_alat",
        "a_min",
        "a_star",
    ]))


def candidate_energy(candidate):
    return as_float_or_none(first_present(candidate, [
        "energy_per_atom",
        "good_energy_per_atom",
        "min_energy_per_atom",
        "candidate_energy_per_atom",
        "e_min",
        "e_star",
    ]))


def candidate_curvature(candidate):
    return as_float_or_none(first_present(candidate, [
        "curvature",
        "good_curvature",
        "minimum_curvature",
        "d2E_da2",
        "second_derivative",
    ]))


def candidate_reason(candidate):
    return first_present(candidate, [
        "reason",
        "reject_reason",
        "status",
        "message",
    ], "")


def candidate_window_arrays(candidate):
    x = as_array(first_present(candidate, [
        "window_alats",
        "alats",
        "final_alats",
        "level2_alats",
    ]))
    y = as_array(first_present(candidate, [
        "window_energy_per_atom",
        "window_energies_per_atom",
        "energies_per_atom",
        "final_energies_per_atom",
        "level2_energies_per_atom",
    ]))

    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

    return x, y


def plot_candidate_windows(ax, candidates, accepted=True):
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        x, y = candidate_window_arrays(candidate)
        if len(x) >= 2:
            x_plot, y_plot = downsample_xy(x, y)
            ax.plot(
                x_plot,
                y_plot,
                linestyle="-" if accepted else "--",
                linewidth=1.0,
                alpha=0.35,
                label="_accepted window" if accepted else "_rejected window",
            )


def scatter_candidates(ax, candidates, accepted=True, annotate=True):
    xs = []
    ys = []
    labels = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        a = candidate_alat(candidate)
        e = candidate_energy(candidate)

        if a is None or e is None:
            # Try deriving from the window if candidate point was not explicitly stored.
            x, y = candidate_window_arrays(candidate)
            if len(x) > 0 and len(y) > 0:
                i = int(np.argmin(y))
                a = float(x[i])
                e = float(y[i])

        if a is None or e is None:
            continue

        xs.append(a)
        ys.append(e)

        if accepted:
            curv = candidate_curvature(candidate)
            if curv is not None:
                labels.append(f"{curv:.3g}")
            else:
                labels.append("")
        else:
            reason = str(candidate_reason(candidate))
            labels.append(reason[:30] if reason else "rejected")

    if not xs:
        return

    if accepted:
        ax.scatter(xs, ys, marker="^", s=65, zorder=7, label="accepted level-2 minima")
    else:
        ax.scatter(xs, ys, marker="x", s=55, zorder=7, label="rejected level-2 minima")

    if annotate:
        for x, y, label in zip(xs, ys, labels):
            if label:
                ax.text(x, y, f" {label}", fontsize=8, va="center")


# ---------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------
def plot_file_record(file_record, output_dir):
    path = file_record["path"]
    specs = file_record["results"]

    if not specs:
        print(f"Skipping {path.name}: no specs in results")
        return

    norm_specs = [normalize_spec(spec, path) for spec in specs]

    nplots = len(norm_specs)
    nrows, ncols = choose_subplot_grid(nplots)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8 * ncols, 6 * nrows),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for ax, spec in zip(axes_flat, norm_specs):
        spec_name = spec["name"]

        all_alats = spec["all_alats"]
        all_energy = spec["all_energy_per_atom"]
        valid_alats = spec["valid_alats"]
        valid_energy = spec["valid_energy_per_atom"]

        equil_alat = spec["equil_alat"]
        good_alat = spec["good_alat"]
        good_energy = spec["good_energy_per_atom"]
        good_curvature = spec["good_curvature"]

        # Main sample clouds/curves
        if SHOW_ALL_SAMPLES and len(all_alats) > 0:
            x_plot, y_plot = downsample_xy(all_alats, all_energy)
            ax.plot(
                x_plot,
                y_plot,
                linewidth=1.0,
                alpha=0.75,
                label="all sampled energy per atom",
            )

        if SHOW_VALID_SAMPLES and len(valid_alats) > 0:
            x_plot, y_plot = downsample_xy(valid_alats, valid_energy)
            ax.scatter(
                x_plot,
                y_plot,
                marker=".",
                s=18,
                alpha=0.65,
                label="valid/selected samples",
            )

        # Old-output candidate minima
        min_indices = safe_indices(spec["min_indices"], len(valid_alats))
        if min_indices:
            ax.scatter(
                valid_alats[min_indices],
                valid_energy[min_indices],
                s=35,
                zorder=5,
                alpha=0.75,
                label="old candidate minima",
            )

            if ANNOTATE_CURVATURE and len(all_alats) > 0:
                for idx in min_indices:
                    a = valid_alats[idx]
                    i_all = nearest_index(all_alats, a)
                    curv = finite_difference_curvature(all_alats, all_energy, i_all)
                    if curv is not None:
                        ax.text(a, valid_energy[idx], f" {curv:.3g}", fontsize=8)

        old_min_index = spec["min_index"]
        try:
            old_min_index = int(old_min_index)
        except Exception:
            old_min_index = None

        if (
            old_min_index is not None
            and 0 <= old_min_index < len(valid_alats)
        ):
            ax.scatter(
                [valid_alats[old_min_index]],
                [valid_energy[old_min_index]],
                s=55,
                zorder=6,
                label="old selected minimum",
            )

        # New multi-resolution diagnostics: coarse and level-1 minima
        if SHOW_LEVEL_MARKERS:
            for a in spec["coarse_min_alats"]:
                if np.isfinite(a):
                    ax.axvline(
                        float(a),
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.45,
                        label="_coarse minima",
                    )

            for a in spec["level1_min_alats"]:
                if np.isfinite(a):
                    ax.axvline(
                        float(a),
                        linestyle="-.",
                        linewidth=1.0,
                        alpha=0.45,
                        label="_level-1 minima",
                    )

            # Add one visible legend entry for the marker types.
            if len(spec["coarse_min_alats"]) > 0:
                ax.plot([], [], linestyle=":", linewidth=1.0, label="coarse minima")
            if len(spec["level1_min_alats"]) > 0:
                ax.plot([], [], linestyle="-.", linewidth=1.0, label="level-1 minima")

        accepted = spec["accepted_candidates"]
        rejected = spec["rejected_candidates"]

        if SHOW_FINAL_WINDOWS:
            plot_candidate_windows(ax, accepted, accepted=True)
            plot_candidate_windows(ax, rejected, accepted=False)

        if SHOW_ACCEPTED_CANDIDATES:
            scatter_candidates(ax, accepted, accepted=True, annotate=ANNOTATE_CURVATURE)

        if SHOW_REJECTED_CANDIDATES:
            scatter_candidates(ax, rejected, accepted=False, annotate=False)

        # Final selected result from new output
        if good_alat is not None:
            if good_energy is None:
                # Derive a y-location from nearest plotted data.
                if len(valid_alats) > 0:
                    i = nearest_index(valid_alats, good_alat)
                    good_energy = float(valid_energy[i])
                elif len(all_alats) > 0:
                    i = nearest_index(all_alats, good_alat)
                    good_energy = float(all_energy[i])

            if good_energy is not None:
                ax.scatter(
                    [good_alat],
                    [good_energy],
                    marker="*",
                    s=180,
                    zorder=9,
                    label="selected best-curvature minimum",
                )

        # Equilibrium reference line and textbox
        if equil_alat is not None and equil_alat != -1:
            ax.axvline(
                equil_alat,
                linestyle="--",
                linewidth=1.2,
                label="reference equilibrium alat",
            )

        textbox_lines = []

        if equil_alat is not None and good_alat is not None and equil_alat != 0:
            pct_error = 100.0 * (1.0 - good_alat / equil_alat)
            textbox_lines.append(f"error = {pct_error:.3g}%")

        if equil_alat is not None:
            textbox_lines.append(f"equil = {equil_alat:.5g}")

        if good_alat is not None:
            textbox_lines.append(f"good = {good_alat:.5g}")

        if good_curvature is not None:
            textbox_lines.append(f"curv = {good_curvature:.4g}")

        if accepted:
            textbox_lines.append(f"accepted = {len(accepted)}")
        if rejected:
            textbox_lines.append(f"rejected = {len(rejected)}")

        strategy = spec["search_strategy"]
        if isinstance(strategy, dict):
            coarse_da = first_present(strategy, ["coarse_del_a", "level0_del_a"])
            fine_da = first_present(strategy, ["level1_del_a", "fine_del_a"])
            final_da = first_present(strategy, ["level2_del_a", "final_del_a"])
            led_order = first_present(strategy, ["led_order"])
            if coarse_da is not None or fine_da is not None or final_da is not None:
                textbox_lines.append(
                    f"Δa = {coarse_da}/{fine_da}/{final_da}"
                )
            if led_order is not None:
                textbox_lines.append(f"LED order = {led_order}")

        if textbox_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(textbox_lines),
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
        ax.set_xlabel("alat")
        ax.set_ylabel("energy per atom")
        ax.grid(True, alpha=0.3)

        # Remove duplicate legend labels while preserving order.
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if not l or l.startswith("_"):
                continue
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        ax.legend(unique_handles, unique_labels, fontsize=8)

    for ax in axes_flat[nplots:]:
        ax.set_visible(False)

    avg_equil_alat = file_record["avg_equil_alat"]
    avg_good_alat = file_record["avg_good_alat"]

    if avg_equil_alat is not None and avg_good_alat is not None and avg_equil_alat != 0:
        avg_diff = 100.0 * (1.0 - avg_good_alat / avg_equil_alat)
        title_tail = (
            f"avg error = {avg_diff:.3g}% "
            f"({avg_equil_alat:.5g} → {avg_good_alat:.5g})"
        )
    else:
        title_tail = "avg equilibrium/good alat unavailable"

    fig.suptitle(f"{path.name}\n{title_tail}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_multires_spec_energy.png"

    if SAVE_FIGURES:
        fig.savefig(output_path, dpi=300)
        print(f"Saved {output_path}")

    if SHOW_FIGURES:
        try:
            plt.gcf().canvas.manager.set_window_title(f"{path.stem}_multires_spec_energy")
        except Exception:
            pass
        plt.show()
    else:
        plt.close(fig)


def main():
    file_records = load_all_files(PLOTDATA_DIR)

    for file_record in file_records:
        plot_file_record(file_record, OUTPUT_DIR)


if __name__ == "__main__":
    main()
