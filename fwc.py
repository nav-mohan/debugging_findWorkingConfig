from kim_tools.ase import *
from typing import Union

import itertools
import logging
import math
import random

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.kim.kim import KIM
from ase.data import chemical_symbols
from ase.lattice.cubic import FaceCenteredCubic

################################################################################
import multiprocessing as mp
import traceback
import signal

NCELLS_PER_SIDE = 1

def _energy_worker(model_name : str, species : list, alat:float,queue):
    """
    worker process. this process is allowed to crash. 
    parent process will survive
    """
    try:
        energy_config = generate_fcc_compute_energy(model=model_name,species=species,alat=alat)
        if energy_config == None:
            queue.put({
                "ok":False,
                "energy" : None,
                "ncells" : None,
                "error":None,
            })

        else:
            pe,ncells = energy_config
            queue.put({
                "ok":True,
                "energy" : pe,
                "ncells" : ncells,
                "error":None,
            })
    except RuntimeError as e :
        queue.put({
            "ok":False,
            "energy" : None,
            "ncells" : None,
            "error":e
        })
    except Exception:
        queue.put({
            "ok":False,
            "energy" : None,
            "ncells" : None,
            "error":traceback.format_exc(),
        })

def generate_fcc_compute_energy_safe(model:str,species:list, alat:float,timeout:float=300.0): # timeout after 5 minutes
    """
    Run generate_fcc_compute_energy in a child process so segfaults 
    do not kill the parent process
    
    Returns 
        (energy,ncells) on success
        None on Python exception, timeout, or segfaults
    """

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    proc = ctx.Process(
        target = _energy_worker,
        args = (model,species,alat,queue)
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"\t\tTIMEOUT at alat = {alat}")
        return None
    
    exitcode = proc.exitcode

    if exitcode != 0:
        if exitcode < 0:
            sig = -exitcode
            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = f"signal {sig}"
            
            print(f"\t\tCRASH at alat = {alat}: child died with {sig_name}")
        else:
            print(f"\t\tFAIL at alat = {alat}: child exit code {exitcode}")

        return None
    
    if queue.empty():
        print(f"\t\tFAIL at alat = {alat}: child exited but returned no result")
        return None
    
    result = queue.get()

    if result["ok"] == False:
        print(f"\t\tPYTHON ERROR at alat = {alat}")
        print(result["error"])
        return None
    
    return (result["energy"],result["ncells"])

def generate_fcc_compute_energy(
    model: Union[str, Calculator],
    species: str,
    alat: float,
) -> tuple[float, int]:
    """
    Construct an FCC lattice large enough to accommodate all species,
    evaluate its energy, and return the energy and size of the supercell.

    Args:
        model: The model name or calculator object.
        species: atomic species to be incorporated into the FCC lattice.
        alat: The lattice constant for the FCC lattice.

    Returns:
        A tuple containing:
            - The total potential energy of the lattice.
            - The number of unit cells along each side of the supercell.
        None if model fails to compute potential-energy
    """

    print(f"\tgenerate_fcc_compute_energy {alat}")
    ncells_per_side = NCELLS_PER_SIDE
    atoms = FaceCenteredCubic(
        size=(ncells_per_side, ncells_per_side, ncells_per_side),
        latticeconstant=alat,
        symbol=species,
        # pbc=False,
        pbc=True,
    )
    if isinstance(model, str):
        calc = KIM(model)
    else:
        calc = model
    atoms.set_calculator(calc)
    print(f"\t\tcalculator set {alat},{ncells_per_side}")
    # compute energy
    try:
        print(f"\t\ttry get_potential_energy {alat},{ncells_per_side}")
        pe = atoms.get_potential_energy()
        print(f"\t\tsuccess get_potential_energy {alat},{ncells_per_side}")
        # General clean-up
        if hasattr(calc, "clean"):
            print(f"\t\tcalc.clean {alat},{ncells_per_side}")
            calc.clean()
        if hasattr(calc, "__del__"):
            print(f"\t\tcalc.del {alat},{ncells_per_side}")
            calc.__del__()
        return pe, ncells_per_side
    except Exception as e:
        print(e)
        return None


################################################################################
def fcc_atoms_in_supercell(n_cells_per_side: int) -> int:
    """
    Compute the number of atoms in an FCC supercell.

    This function calculates the total number of atoms in a Face-Centered Cubic (FCC)
    supercell based on the number of unit cells along each side. In an FCC structure,
    each unit cell contains 4 atoms.

    Args:
        n_cells_per_side: Number of unit cells along each side of the supercell.
                                This represents the size of the supercell.

    Returns:
        The total number of atoms in the FCC supercell.
    """

    atoms_per_unit_cell = 4
    total_unit_cells = n_cells_per_side**3
    total_atoms = total_unit_cells * atoms_per_unit_cell
    return (int)(total_atoms)


################################################################################
def _validate_led_order(order: int) -> None:
    """Validate supported Local Edge Detection stencil orders."""

    supported_orders = {3, 5, 7, 9}
    if order not in supported_orders:
        raise ValueError(
            f"Unsupported LED order {order}. "
            f"Supported orders are {sorted(supported_orders)}."
        )


def _led_start_index(order: int) -> int:
    """Return the first alat/y index that has a defined LED value."""

    _validate_led_order(order)
    return (order - 1) // 2


def _led_stop_index_exclusive(n_points: int, order: int) -> int:
    """Return the exclusive stop index for alat/y indices with defined LED values."""

    _validate_led_order(order)
    return n_points - ((order + 1) // 2)


def _led_coefficients(order: int) -> list[float]:
    """Return binomial finite-difference coefficients for the LED stencil."""

    _validate_led_order(order)

    # order=5 gives the old stencil:
    # [-1, 5, -10, 10, -5, 1] / 6
    normalization = float(order + 1)
    return [
        ((-1) ** (k + 1)) * math.comb(order, k) / normalization
        for k in range(order + 1)
    ]


def local_edge_detection(x: list, y: list, order: int = 5) -> list:
    """
    Computes the Local Edge Detection (LED) values for the x-y curve.
    The algorithm helps to identify discontinuities or abrupt changes in the curve.

    Based on the paper:
    A. Gelb and E. Tadmor, "Local edge detection for non-linear signals,"
    Journal of Scientific Computing, 28:279-306, 2006.

    Args:
        x: A list of x-values representing the independent variable of the curve.
           This function assumes the x-values are ordered and approximately uniform.
        y: A list of y-values representing the dependent variable of the curve.
        order: LED stencil order. Supported values are 3, 5, 7, and 9.

    Returns:
        A list of LED values measuring the strength of discontinuities in the
        x-y curve.

    Notes:
        For a given order, LED[k] corresponds to the original point
        x[k + start_idx], where start_idx = (order - 1) // 2.

        Examples:
            order=3: LED[0] corresponds to x[1]
            order=5: LED[0] corresponds to x[2]
            order=7: LED[0] corresponds to x[3]
            order=9: LED[0] corresponds to x[4]
    """

    _validate_led_order(order)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    led_values = []
    n_points = len(y)
    start_idx = _led_start_index(order)
    stop_idx = _led_stop_index_exclusive(n_points, order)

    if stop_idx <= start_idx:
        return led_values

    coeffs = _led_coefficients(order)
    left_radius = (order - 1) // 2

    for j in range(start_idx, stop_idx):
        # The stencil covers y[j-left_radius] ... y[j-left_radius+order].
        # For order=5 this reproduces the previous formula using
        # y[j-2] ... y[j+3].
        stencil_start = j - left_radius
        led = 0.0
        for k, coeff in enumerate(coeffs):
            led += coeff * y[stencil_start + k]

        led_values.append(led)

    return led_values


################################################################################
def filter_good_alat(
    alats: list,
    energies_per_atom: list,
    leds: list,
    etol: list = [5e-2, 5e2],
    led_tol: float = 1.0,
    led_order: int = 5,
) -> dict:
    """
    Filter a good lattice constant (alat) based on energy and LED criteria.

    Args:
        alats: A list of lattice parameters.
        energies_per_atom: Energy-per-atom values corresponding to each alat.
        leds: LED values computed by local_edge_detection(..., order=led_order).
        etol: Minimum and maximum allowed absolute energy per atom.
        led_tol: Edge threshold. If abs(LED) > led_tol, an edge is detected.
        led_order: LED stencil order. Supported values are 3, 5, 7, and 9.

    Edge-discard rule:
        If an edge is detected at original index i, discard all points from
        i - led_order through i + led_order, inclusive.
    """

    _validate_led_order(led_order)

    N = len(alats)
    if len(energies_per_atom) != N:
        raise ValueError("alats and energies_per_atom must have the same length")

    start_idx = _led_start_index(led_order)
    end_idx = _led_stop_index_exclusive(N, led_order)

    expected_led_count = max(0, end_idx - start_idx)
    if len(leds) != expected_led_count:
        raise ValueError(
            f"Expected {expected_led_count} LED values for N={N} and "
            f"led_order={led_order}, but got {len(leds)}."
        )

    edge_indices = []
    discarded_indices = set()

    for led_idx, led in enumerate(leds):
        original_idx = start_idx + led_idx
        if abs(led) > led_tol:
            edge_indices.append(original_idx)

            discard_start = max(0, original_idx - led_order)
            discard_stop = min(N - 1, original_idx + led_order)
            discarded_indices.update(range(discard_start, discard_stop + 1))

    valid_energy_per_atom = []
    valid_leds = []
    valid_alats = []
    valid_original_indices = []

    for i in range(start_idx, end_idx):
        if i in discarded_indices:
            continue

        alat = alats[i]
        energy_per_atom = energies_per_atom[i]
        led = leds[i - start_idx]

        if abs(energy_per_atom) > etol[1] or abs(energy_per_atom) < etol[0]:
            continue

        valid_leds.append(led)
        valid_energy_per_atom.append(energy_per_atom)
        valid_alats.append(alat)
        valid_original_indices.append(i)

    if len(valid_alats) == 0:
        return {
            "good_alat": -1.0,
            "min_led": -1.0,
            "good_ncells": -1,
            "valid_alats": [-1.0],
            "valid_energy_per_atom": [0],
            "indices": [0],
            "min_index": 0,
            "all_alats": alats,
            "all_energies_per_atom": energies_per_atom,
            "led_order": led_order,
            "edge_indices": edge_indices,
            "discarded_indices": sorted(discarded_indices),
            "valid_original_indices": [],
        }

    valid_leds = np.array(valid_leds).tolist()
    valid_energy_per_atom = np.array(valid_energy_per_atom).tolist()
    valid_alats = np.array(valid_alats).tolist()

    # Now pick a local minimum from the valid energy-per-atom values.
    # Note: these indices are relative to valid_alats/valid_energy_per_atom.
    from scipy.signal import find_peaks

    indices, _ = find_peaks(-np.array(valid_energy_per_atom))
    indices = indices.tolist()

    if len(indices) == 0:
        min_index = int(np.argmin(valid_energy_per_atom))
    else:
        min_index = min(indices, key=lambda i: valid_energy_per_atom[i])

    good_alat = valid_alats[min_index]
    min_led = valid_leds[min_index]

    return {
        "good_alat": good_alat,
        "min_led": min_led,
        "good_ncells": NCELLS_PER_SIDE,
        "valid_alats": valid_alats,
        "valid_energy_per_atom": valid_energy_per_atom,
        "indices": indices,
        "min_index": min_index,
        "min_original_index": valid_original_indices[min_index],
        "all_alats": alats,
        "all_energies_per_atom": energies_per_atom,
        "led_order": led_order,
        "edge_indices": edge_indices,
        "discarded_indices": sorted(discarded_indices),
        "valid_original_indices": valid_original_indices,
    }


################################################################################

from ase.data import atomic_numbers, covalent_radii
import kimpy
def query_kim_influence_distance(model_name: str) -> float:
    """
    Return the KIM model influence distance.

    This is the model-reported interaction range, often what people loosely
    call the model cutoff.
    """
    units_accepted, kim_model = kimpy.model.create(
        kimpy.numbering.zeroBased,
        kimpy.length_unit.A,
        kimpy.energy_unit.eV,
        kimpy.charge_unit.e,
        kimpy.temperature_unit.K,
        kimpy.time_unit.ps,
        model_name,
    )

    try:
        r_infl = kim_model.get_influence_distance()
        return float(r_infl)
    finally:
        # Depending on kimpy/KIM API version, explicit cleanup may differ.
        # If your kim_model object has clean/destroy behavior, call it here.
        if hasattr(kim_model, "destroy"):
            kim_model.destroy()



# -----------------------------------------------------------------------------
# Multi-resolution FCC lattice-constant search helpers
# -----------------------------------------------------------------------------

def _round_alat_key(alat: float, ndigits: int = 8) -> float:
    """Round alat for use as a cache key."""

    return round(float(alat), ndigits)


def _alat_grid(a_min: float, a_max: float, del_a: float) -> list[float]:
    """
    Build an inclusive, uniformly spaced alat grid.

    Uses integer stepping to avoid accumulating floating-point error.
    """

    if del_a <= 0:
        raise ValueError("del_a must be positive")

    a_min = float(a_min)
    a_max = float(a_max)
    if a_max < a_min:
        return []

    n_steps = int(math.floor((a_max - a_min) / del_a + 1.0e-12))
    grid = [a_min + i * del_a for i in range(n_steps + 1)]

    # Include the right endpoint if it is not already represented.
    if len(grid) == 0 or grid[-1] < a_max - 0.5 * del_a:
        grid.append(a_max)

    return [_round_alat_key(a) for a in grid]


def _compute_energy_per_atom_cached(
    model: Union[str, Calculator],
    species: str,
    alat: float,
    energy_cache: dict,
    use_safe: bool = True,
) -> Union[tuple[float, int], None]:
    """
    Compute energy per atom at one alat, using a cache to avoid duplicate calls.

    Returns:
        (energy_per_atom, ncells) on success, None on failure.
    """

    key = _round_alat_key(alat)
    if key in energy_cache:
        return energy_cache[key]

    if use_safe:
        val = generate_fcc_compute_energy_safe(model, species, key)
    else:
        val = generate_fcc_compute_energy(model, species, key)

    if val is None:
        energy_cache[key] = None
        return None

    energy_total, ncell = val
    natoms = fcc_atoms_in_supercell(ncell)
    energy_pa = float(energy_total) / float(natoms)

    energy_cache[key] = (energy_pa, int(ncell))
    return energy_cache[key]


def _scan_alat_range(
    model: Union[str, Calculator],
    species: str,
    a_min: float,
    a_max: float,
    del_a: float,
    energy_cache: dict,
    use_safe: bool = True,
    switch_to_unsafe_after: Union[int, None] = None,
    early_stop_plateau: bool = False,
    min_scan_alat: Union[float, None] = None,
    plateau_window: int = 20,
    plateau_slope_tol: float = 1.0e-3,
    plateau_range_tol: float = 5.0e-4,
) -> dict:
    """
    Scan an alat interval and return successful energy-per-atom evaluations.

    If use_safe=True and switch_to_unsafe_after is a positive integer, the
    scan starts with subprocess-protected evaluations and switches to direct
    evaluations after that many consecutive successful safe evaluations.

    If early_stop_plateau=True, stop once the most recent successful
    energy-per-atom evaluations have plateaued. This is mainly useful for
    the level-0 full scan, where scanning all the way to a large amax can
    be wasted once the curve has reached the asymptotic region.
    """

    alats = []
    energies_per_atom = []
    ncells = []

    current_use_safe = bool(use_safe)
    consecutive_safe_successes = 0
    switched_to_unsafe = False
    switch_alat = None

    if switch_to_unsafe_after is not None and switch_to_unsafe_after <= 0:
        current_use_safe = False
        switched_to_unsafe = bool(use_safe)
        switch_to_unsafe_after = 0

    for a in _alat_grid(a_min, a_max, del_a):
        used_safe_for_this_point = current_use_safe

        try:
            result = _compute_energy_per_atom_cached(
                model=model,
                species=species,
                alat=a,
                energy_cache=energy_cache,
                use_safe=current_use_safe,
            )
        except Exception as e:
            print(f"		scan exception at alat={a}: {e}")
            if used_safe_for_this_point:
                consecutive_safe_successes = 0
            continue

        if result is None:
            if used_safe_for_this_point:
                consecutive_safe_successes = 0
            continue

        energy_pa, ncell = result
        if not np.isfinite(energy_pa):
            if used_safe_for_this_point:
                consecutive_safe_successes = 0
            continue

        alats.append(float(a))
        energies_per_atom.append(float(energy_pa))
        ncells.append(int(ncell))
        mode = "safe" if used_safe_for_this_point else "direct"
        print(f"		alat = {a} | energy_per_atom = {energy_pa} | mode = {mode}")

        if used_safe_for_this_point:
            consecutive_safe_successes += 1
            if (
                switch_to_unsafe_after is not None
                and switch_to_unsafe_after > 0
                and consecutive_safe_successes >= switch_to_unsafe_after
            ):
                current_use_safe = False
                switched_to_unsafe = True
                switch_alat = float(a)
                print(
                    f"		Switching scan to direct execution after "
                    f"{consecutive_safe_successes} consecutive safe successes "
                    f"near alat = {a}"
                )

        if early_stop_plateau:
            can_check_plateau = True
            if min_scan_alat is not None and float(a) < float(min_scan_alat):
                can_check_plateau = False

            if can_check_plateau and energy_plateau_detected(
                alats,
                energies_per_atom,
                window=plateau_window,
                slope_tol=plateau_slope_tol,
                range_tol=plateau_range_tol,
            ):
                print(f"		Early stopping scan: energy plateau detected near alat = {a}")
                break

    return {
        "alats": alats,
        "energies_per_atom": energies_per_atom,
        "ncells": ncells,
        "del_a": float(del_a),
        "a_min": float(a_min),
        "a_max": float(a_max),
        "initial_use_safe": bool(use_safe),
        "switch_to_unsafe_after": switch_to_unsafe_after,
        "switched_to_unsafe": switched_to_unsafe,
        "switch_alat": switch_alat,
    }

def _merge_scan_points(scans: list[dict]) -> dict:
    """Merge multiple scan dictionaries into one sorted, duplicate-free curve."""

    data = {}
    for scan in scans:
        for a, e, ncell in zip(
            scan.get("alats", []),
            scan.get("energies_per_atom", []),
            scan.get("ncells", []),
        ):
            data[_round_alat_key(a)] = (float(a), float(e), int(ncell))

    merged = [data[k] for k in sorted(data.keys())]
    return {
        "alats": [x[0] for x in merged],
        "energies_per_atom": [x[1] for x in merged],
        "ncells": [x[2] for x in merged],
    }


def _local_minima_indices(
    energies_per_atom: list[float],
    include_global_if_none: bool = True,
) -> list[int]:
    """
    Return local-minimum indices in a sampled energy curve.

    Plateaus are handled by accepting <= on both sides. If no strict/local minima
    are found, the global minimum is returned as a fallback.
    """

    y = np.asarray(energies_per_atom, dtype=float)
    if len(y) == 0:
        return []
    if len(y) == 1:
        return [0]

    indices = []
    for i in range(1, len(y) - 1):
        if not np.isfinite(y[i - 1]) or not np.isfinite(y[i]) or not np.isfinite(y[i + 1]):
            continue
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            # Avoid adding every point in a perfectly flat plateau by keeping
            # the first plateau point only.
            if len(indices) == 0 or i != indices[-1] + 1:
                indices.append(i)

    if len(indices) == 0 and include_global_if_none:
        indices = [int(np.nanargmin(y))]

    return indices


def _dedupe_candidate_alats(candidate_alats: list[float], min_sep: float) -> list[float]:
    """Deduplicate candidate minima by keeping candidates separated by min_sep."""

    out = []
    for a in sorted(float(x) for x in candidate_alats):
        if len(out) == 0 or abs(a - out[-1]) >= min_sep:
            out.append(a)
    return out


def _central_second_derivative(
    alats: list[float],
    energies_per_atom: list[float],
    index: int,
) -> Union[float, None]:
    """
    Estimate d²E/da² at a sampled point using a three-point central formula.

    The simple formula assumes approximately uniform spacing around index.
    """

    if index <= 0 or index >= len(alats) - 1:
        return None

    x = np.asarray(alats, dtype=float)
    y = np.asarray(energies_per_atom, dtype=float)

    h_left = x[index] - x[index - 1]
    h_right = x[index + 1] - x[index]
    if h_left <= 0 or h_right <= 0:
        return None

    if not np.isclose(h_left, h_right, rtol=1.0e-3, atol=1.0e-10):
        return None

    h = 0.5 * (h_left + h_right)
    curvature = (y[index - 1] - 2.0 * y[index] + y[index + 1]) / (h * h)
    return float(curvature)


def _window_is_continuous_by_led(
    alats: list[float],
    energies_per_atom: list[float],
    led_tol: float,
    led_order: int = 5,
    energy_bound: list = [5e-2, 5e2],
) -> dict:
    """
    Apply LED to one fine-sampled window and decide whether it is continuous.

    A window is rejected if any LED value exceeds led_tol or if too few points
    exist to define the LED stencil.
    """

    if len(alats) != len(energies_per_atom):
        raise ValueError("alats and energies_per_atom must have the same length")

    if len(alats) < led_order + 1:
        return {
            "continuous": False,
            "reason": "not enough points for LED stencil",
            "leds": [],
            "edge_indices": [],
            "max_abs_led": np.inf,
        }

    leds = local_edge_detection(alats, energies_per_atom, order=led_order)
    start_idx = _led_start_index(led_order)

    edge_indices = []
    max_abs_led = 0.0
    for led_idx, led in enumerate(leds):
        max_abs_led = max(max_abs_led, abs(float(led)))
        if abs(led) > led_tol:
            edge_indices.append(start_idx + led_idx)

    if len(edge_indices) > 0:
        return {
            "continuous": False,
            "reason": "LED edge detected",
            "leds": leds,
            "edge_indices": edge_indices,
            "max_abs_led": max_abs_led,
        }

    # Also reject windows with non-finite or absurd energies.
    for e in energies_per_atom:
        if not np.isfinite(e):
            return {
                "continuous": False,
                "reason": "non-finite energy",
                "leds": leds,
                "edge_indices": edge_indices,
                "max_abs_led": max_abs_led,
            }
        if abs(e) > energy_bound[1]:
            return {
                "continuous": False,
                "reason": "energy outside upper bound",
                "leds": leds,
                "edge_indices": edge_indices,
                "max_abs_led": max_abs_led,
            }

    return {
        "continuous": True,
        "reason": "continuous",
        "leds": leds,
        "edge_indices": edge_indices,
        "max_abs_led": max_abs_led,
    }


def _best_minimum_in_window_by_curvature(
    alats: list[float],
    energies_per_atom: list[float],
    energy_bound: list = [5e-2, 5e2],
) -> Union[dict, None]:
    """
    Find local minima in a final fine window and return the one with largest
    positive second derivative.
    """

    minima = _local_minima_indices(energies_per_atom, include_global_if_none=True)
    candidates = []

    for idx in minima:
        e = float(energies_per_atom[idx])
        if not np.isfinite(e):
            continue
        if abs(e) > energy_bound[1]:
            continue
        # Keep the lower energy bound as a weak filter, matching the old code.
        if abs(e) < energy_bound[0]:
            continue

        curvature = _central_second_derivative(alats, energies_per_atom, idx)
        if curvature is None or curvature <= 0.0 or not np.isfinite(curvature):
            continue

        candidates.append({
            "index": int(idx),
            "alat": float(alats[idx]),
            "energy_per_atom": e,
            "curvature": float(curvature),
        })

    if len(candidates) == 0:
        return None

    # User requested: pick the best minimum as largest double-derivative.
    return max(candidates, key=lambda d: d["curvature"])


# this is only for mono-species.
# species will be a string, not a list.
def find_working_configuration_FCC(
    model: Union[str, Calculator],
    species: str,
    energy_bound: list = [5e-2, 5e2],
    led_tol: float = 1.0,
    led_order: int = 5,
) -> dict:

    """
    Multi-resolution FCC lattice-constant search.

    Procedure:
        1. Coarse scan with del_a = 0.1 over [amin, amax], with plateau
           early termination after min_scan_alat.
        2. Find minima in the coarse energy-alat curve.
        3. Around each coarse minimum, scan +-0.5 A with del_a = 0.01.
        4. Find refined minima in each level-1 fine curve.
        5. Around each level-1 minimum, scan +-0.05 A with del_a = 0.001.
        6. Run LED on each final fine window.
           - If the final window is discontinuous, reject that minimum.
           - If it is continuous, keep the best local minimum in that window.
        7. Pick the accepted minimum with the largest positive second derivative.

    Returns a dictionary compatible with the older plotting fields, plus detailed
    multiresolution diagnostics.
    """

    _validate_led_order(led_order)

    cov = covalent_radii[atomic_numbers[species]]
    amin = max(np.sqrt(2) * cov, 1.5)
    amax = 12.0

    # For the full level-0 scan, do not consider plateau early stopping until
    # we have scanned far enough to include the physically relevant basins.
    min_scan_alat = 6.5
    plateau_detection_window = 20
    plateau_detection_slope = 1.0e-3
    plateau_detection_range = 5.0e-4
    coarse_switch_to_unsafe_after = 25

    # If this is a KIM model, expand the scan range if the influence distance is
    # larger than the default amax. Sim_LAMMPS models do not use this kimpy path.
    if isinstance(model, str) and model[:3] != "Sim":
        min_cutoff = query_kim_influence_distance(model)
        if min_cutoff > amax:
            amax = 2.0 * min_cutoff
            min_scan_alat = min_cutoff

    coarse_del_a = 0.1
    level1_del_a = 0.01
    level2_del_a = 0.001
    level1_half_width = 0.5
    level2_half_width = 0.05

    # Cache avoids re-evaluating the same alat when windows overlap.
    energy_cache = {}
    all_scans = []

    print("\tStarting level-0 coarse scan")
    coarse_scan = _scan_alat_range(
        model=model,
        species=species,
        a_min=amin,
        a_max=amax,
        del_a=coarse_del_a,
        energy_cache=energy_cache,
        use_safe=True,
        switch_to_unsafe_after=coarse_switch_to_unsafe_after,
        early_stop_plateau=True,
        min_scan_alat=min_scan_alat,
        plateau_window=plateau_detection_window,
        plateau_slope_tol=plateau_detection_slope,
        plateau_range_tol=plateau_detection_range,
    )
    all_scans.append(coarse_scan)

    coarse_min_indices = _local_minima_indices(
        coarse_scan["energies_per_atom"],
        include_global_if_none=True,
    )
    coarse_min_alats = [coarse_scan["alats"][i] for i in coarse_min_indices]
    coarse_min_alats = _dedupe_candidate_alats(coarse_min_alats, min_sep=0.5 * coarse_del_a)

    print(f"\tCoarse minima candidates: {coarse_min_alats}")

    level1_candidates = []
    level1_scans = []

    for coarse_a in coarse_min_alats:
        a1_min = max(amin, coarse_a - level1_half_width)
        a1_max = min(amax, coarse_a + level1_half_width)

        print(f"\tStarting level-1 scan around coarse minimum {coarse_a}: [{a1_min}, {a1_max}]")
        scan1 = _scan_alat_range(
            model=model,
            species=species,
            a_min=a1_min,
            a_max=a1_max,
            del_a=level1_del_a,
            energy_cache=energy_cache,
            # Coarse scan already found this neighborhood to be evaluable.
            # Use direct calls here to avoid spawning hundreds of subprocesses.
            use_safe=False,
        )
        level1_scans.append(scan1)
        all_scans.append(scan1)

        min_indices1 = _local_minima_indices(
            scan1["energies_per_atom"],
            include_global_if_none=True,
        )

        for idx1 in min_indices1:
            level1_candidates.append({
                "coarse_alat": float(coarse_a),
                "level1_index": int(idx1),
                "level1_alat": float(scan1["alats"][idx1]),
                "level1_energy_per_atom": float(scan1["energies_per_atom"][idx1]),
            })

    level1_min_alats = _dedupe_candidate_alats(
        [c["level1_alat"] for c in level1_candidates],
        min_sep=0.5 * level1_del_a,
    )

    print(f"\tLevel-1 minima candidates: {level1_min_alats}")

    accepted_candidates = []
    rejected_candidates = []
    level2_scans = []

    for level1_a in level1_min_alats:
        a2_min = max(amin, level1_a - level2_half_width)
        a2_max = min(amax, level1_a + level2_half_width)

        print(f"\tStarting level-2 scan around level-1 minimum {level1_a}: [{a2_min}, {a2_max}]")
        scan2 = _scan_alat_range(
            model=model,
            species=species,
            a_min=a2_min,
            a_max=a2_max,
            del_a=level2_del_a,
            energy_cache=energy_cache,
            # Level-1 scan already refined this neighborhood.
            # Use direct calls for the dense level-2 scan.
            use_safe=False,
        )
        level2_scans.append(scan2)
        all_scans.append(scan2)

        led_info = _window_is_continuous_by_led(
            scan2["alats"],
            scan2["energies_per_atom"],
            led_tol=led_tol,
            led_order=led_order,
            energy_bound=energy_bound,
        )

        if not led_info["continuous"]:
            rejected_candidates.append({
                "level1_alat": float(level1_a),
                "reason": led_info["reason"],
                "edge_indices": led_info["edge_indices"],
                "max_abs_led": led_info["max_abs_led"],
                "window_alats": scan2["alats"],
                "window_energies_per_atom": scan2["energies_per_atom"],
            })
            print(f"\tRejected level-2 window around {level1_a}: {led_info['reason']}")
            continue

        best_in_window = _best_minimum_in_window_by_curvature(
            scan2["alats"],
            scan2["energies_per_atom"],
            energy_bound=energy_bound,
        )

        if best_in_window is None:
            rejected_candidates.append({
                "level1_alat": float(level1_a),
                "reason": "no positive-curvature minimum in continuous window",
                "edge_indices": led_info["edge_indices"],
                "max_abs_led": led_info["max_abs_led"],
                "window_alats": scan2["alats"],
                "window_energies_per_atom": scan2["energies_per_atom"],
            })
            print(f"\tRejected level-2 window around {level1_a}: no positive-curvature minimum")
            continue

        accepted_candidates.append({
            "level1_alat": float(level1_a),
            "best": best_in_window,
            "max_abs_led": led_info["max_abs_led"],
            "window_alats": scan2["alats"],
            "window_energies_per_atom": scan2["energies_per_atom"],
            "window_leds": led_info["leds"],
        })
        print(
            f"\tAccepted minimum alat={best_in_window['alat']} "
            f"curvature={best_in_window['curvature']}"
        )

    merged = _merge_scan_points(all_scans)

    if len(accepted_candidates) == 0:
        return {
            "good_alat": -1.0,
            "min_led": -1.0,
            "good_ncells": -1,
            "valid_alats": [-1.0],
            "valid_energy_per_atom": [0],
            "indices": [0],
            "min_index": 0,
            "all_alats": merged["alats"],
            "all_energies_per_atom": merged["energies_per_atom"],
            "led_order": led_order,
            "coarse_min_alats": coarse_min_alats,
            "level1_min_alats": level1_min_alats,
            "accepted_candidates": [],
            "rejected_candidates": rejected_candidates,
            "search_strategy": "multiresolution_curvature_led",
            "plateau_early_stop": {
                "enabled_on_level0": True,
                "min_scan_alat": float(min_scan_alat),
                "window": int(plateau_detection_window),
                "slope_tol": float(plateau_detection_slope),
                "range_tol": float(plateau_detection_range),
            },
            "coarse_safe_to_direct_switch": {
                "enabled": True,
                "after_consecutive_safe_successes": int(coarse_switch_to_unsafe_after),
                "switched_to_unsafe": bool(coarse_scan.get("switched_to_unsafe", False)),
                "switch_alat": coarse_scan.get("switch_alat"),
            },
        }

    # User requested final selection by largest double derivative.
    best_candidate = max(
        accepted_candidates,
        key=lambda c: c["best"]["curvature"],
    )

    best = best_candidate["best"]
    valid_alats = best_candidate["window_alats"]
    valid_energy_per_atom = best_candidate["window_energies_per_atom"]
    min_index = int(best["index"])

    # In the accepted final window there are no LED violations. Report the max
    # abs LED of that window as min_led diagnostic.
    min_led = float(best_candidate["max_abs_led"])

    # Choose ncells from the cached value at best alat.
    best_cached = energy_cache.get(_round_alat_key(best["alat"]))
    good_ncells = best_cached[1] if best_cached is not None else NCELLS_PER_SIDE

    return {
        "good_alat": float(best["alat"]),
        "good_energy_per_atom": float(best["energy_per_atom"]),
        "good_curvature": float(best["curvature"]),
        "min_led": min_led,
        "good_ncells": int(good_ncells),
        "valid_alats": valid_alats,
        "valid_energy_per_atom": valid_energy_per_atom,
        "indices": _local_minima_indices(valid_energy_per_atom, include_global_if_none=False),
        "min_index": min_index,
        "all_alats": merged["alats"],
        "all_energies_per_atom": merged["energies_per_atom"],
        "led_order": led_order,
        "coarse_min_alats": coarse_min_alats,
        "level1_min_alats": level1_min_alats,
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": rejected_candidates,
        "search_strategy": "multiresolution_curvature_led",
        "plateau_early_stop": {
            "enabled_on_level0": True,
            "min_scan_alat": float(min_scan_alat),
            "window": int(plateau_detection_window),
            "slope_tol": float(plateau_detection_slope),
            "range_tol": float(plateau_detection_range),
        },
        "coarse_safe_to_direct_switch": {
            "enabled": True,
            "after_consecutive_safe_successes": int(coarse_switch_to_unsafe_after),
            "switched_to_unsafe": bool(coarse_scan.get("switched_to_unsafe", False)),
            "switch_alat": coarse_scan.get("switch_alat"),
        },
    }


def energy_plateau_detected(
    alats,
    energies_per_atom,
    window=8,
    slope_tol=1e-3,
    range_tol=1e-3,
):
    """
    Detect whether the last `window` energy-per-atom values have plateaued.

    slope_tol:
        maximum allowed approximate slope, in eV / atom / Angstrom

    range_tol:
        maximum allowed energy variation over the window, in eV / atom
    """

    if len(energies_per_atom) < window:
        return False

    x = np.array(alats[-window:], dtype=float)
    y = np.array(energies_per_atom[-window:], dtype=float)

    if not np.all(np.isfinite(y)):
        return False

    # Energy variation over the recent window
    recent_range = np.max(y) - np.min(y)

    # Linear slope over the recent window
    slope, intercept = np.polyfit(x, y, deg=1)

    if abs(slope) < slope_tol and recent_range < range_tol:
        return True

    return False

