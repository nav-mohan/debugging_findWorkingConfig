from __future__ import annotations

import math
import multiprocessing as mp
import signal
import traceback

import numpy as np
import kim_tools.ase as kim_ase_utils
from ase.calculators.kim.kim import KIM
from ase.data import atomic_numbers, covalent_radii
from ase.lattice.cubic import FaceCenteredCubic

import kimpy


__all__ = ["find_equilibrium_config_FCC"]

FWC_NCELLS_PER_SIDE = 2


def _species_label(species_list: list[str]) -> str:
    return "-".join(species_list)


def make_fcc_template(ncells_per_side: int, species_list: list[str]):
    """
    Create a generic FCC template large enough to contain at least
    len(species_list) atoms. The actual species are assigned later.
    """

    while True:
        atoms = FaceCenteredCubic(
            size=(ncells_per_side, ncells_per_side, ncells_per_side),
            latticeconstant=1.0,
            symbol="H",
            pbc=True,
        )

        if len(atoms) < len(species_list):
            ncells_per_side += 1
        else:
            break

    print(
        f"		#atoms = {len(atoms)}, ncells = {ncells_per_side}, "
        f"species_list = {species_list}"
    )
    return atoms, ncells_per_side


def fcc_atoms_in_supercell(ncells_per_side: int) -> int:
    return int(4 * ncells_per_side**3)


def generate_fcc_compute_energy(
    model: str,
    species_list: list[str],
    alat: float,
) -> tuple[float, int] | None:
    """
    Return (total_energy, ncells_per_side) for an FCC configuration.
    Returns None if the model raises a Python-level exception.
    """

    label = _species_label(species_list)
    print(f"\tgenerate_fcc_compute_energy label={label} alat={alat}")

    atoms, actual_ncells_per_side = make_fcc_template(ncells_per_side=FWC_NCELLS_PER_SIDE,species_list=species_list,)
    atoms.set_cell([actual_ncells_per_side * float(alat)] * 3,scale_atoms=True,)
    kim_ase_utils.randomize_species(atoms, species_list)

    calc = KIM(model)

    atoms.calc = calc

    try:
        pe = atoms.get_potential_energy()
        return float(pe), actual_ncells_per_side
    except Exception as e:
        print(f"\t\tenergy exception label={label} alat={alat}: {e}")
        return None
    finally:
        try:
            if hasattr(calc, "clean"):
                calc.clean()
        except Exception:
            pass
        try:
            if hasattr(calc, "__del__"):
                calc.__del__()
        except Exception:
            pass


def _energy_worker(model_name: str, species_list: list[str], alat: float, queue):
    """Child-process energy worker. The child may crash; the parent survives."""

    try:
        energy_config = generate_fcc_compute_energy(model=model_name,species_list=species_list,alat=alat,)
        if energy_config is None:
            queue.put({"ok": False, "energy": None, "ncells": None, "error": None})
        else:
            energy, ncells = energy_config
            queue.put({"ok": True, "energy": energy, "ncells": ncells, "error": None})
    except Exception:
        queue.put({"ok": False,"energy": None,"ncells": None,"error": traceback.format_exc(),})


def generate_fcc_compute_energy_safe(
    model: str,
    species_list: list[str],
    alat: float,
    timeout: float = 300.0,
) -> tuple[float, int] | None:
    """Run one energy evaluation in a child process."""

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_energy_worker, args=(model, species_list, alat, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"\t\tTIMEOUT at alat={alat}")
        return None

    if proc.exitcode != 0:
        if proc.exitcode < 0:
            sig = -proc.exitcode
            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = f"signal {sig}"
            print(f"\t\tCRASH at alat={alat}: child died with {sig_name}")
        else:
            print(f"\t\tFAIL at alat={alat}: child exit code {proc.exitcode}")
        return None

    if queue.empty():
        print(f"\t\tFAIL at alat={alat}: child exited but returned no result")
        return None

    result = queue.get()
    if not result.get("ok", False):
        print(f"\t\tPYTHON ERROR at alat={alat}")
        print(result.get("error"))
        return None

    return float(result["energy"]), int(result["ncells"])


def _round_alat(alat: float) -> float:
    return round(float(alat), 8)


def _compute_energy_per_atom_cached(model: str,species_list: list[str],alat: float,energy_cache: dict,use_safe: bool,timeout: float = 300.0,) -> dict | None:
    cache_key = (tuple(species_list), _round_alat(alat))
    if cache_key in energy_cache: return energy_cache[cache_key]

    if use_safe:
        val = generate_fcc_compute_energy_safe(model=model,species_list=species_list,alat=float(alat),timeout=timeout,)
    else:
        val = generate_fcc_compute_energy(model=model,species_list=species_list,alat=float(alat),)

    if val is None:
        energy_cache[cache_key] = None
        return None

    energy_total, ncells = val
    energy_per_atom = float(energy_total) / float(fcc_atoms_in_supercell(ncells))
    result = {
        "alat": float(alat),
        "energy_total": float(energy_total),
        "energy_per_atom": float(energy_per_atom),
        "ncells": int(ncells),
    }
    energy_cache[cache_key] = result
    return result


def energy_plateau_detected(
    alats: list[float],
    energies_per_atom: list[float],
    window: int = 20,
    slope_tol: float = 1.0e-3,
    range_tol: float = 5.0e-4,
) -> bool:
    if len(energies_per_atom) < window:
        return False

    x = np.asarray(alats[-window:], dtype=float)
    y = np.asarray(energies_per_atom[-window:], dtype=float)
    if not np.all(np.isfinite(y)):
        return False

    recent_range = float(np.max(y) - np.min(y))
    slope, _ = np.polyfit(x, y, deg=1)
    return abs(float(slope)) < slope_tol and recent_range < range_tol

def _scan_alat_range(
    model: str,
    species_list: list[str],
    a_min: float,
    a_max: float,
    del_a: float,
    energy_cache: dict,
    use_safe: bool = True,
    safe_successes_before_direct: int = 10,
    timeout: float = 300.0,
    early_stop_plateau: bool = True,
    min_scan_alat: float | None = None,
    plateau_window: int = 20,
    plateau_slope_tol: float = 1.0e-3,
    plateau_range_tol: float = 5.0e-4,
) -> dict:
    """
    Coarse lattice sweep. If use_safe=True, start with subprocess evaluations
    and switch to direct execution after safe_successes_before_direct consecutive
    successful evaluations.
    """

    a_min = float(a_min)
    a_max = float(a_max)
    del_a = float(del_a)
    if a_max < a_min: a_max,a_min = a_min,a_max

    n_steps = int(math.floor((a_max - a_min) / del_a + 1.0e-9))

    current_use_safe = bool(use_safe)
    safe_success_count = 0
    switched_to_direct = False
    switch_alat = None
    plateau_stop_alat = None

    alats = []
    energies_per_atom = []
    energy_total = []
    ncells = []

    for j in range(n_steps + 1):
        alat = _round_alat(a_min + j * del_a)
        try:
            row = _compute_energy_per_atom_cached(
                model=model,
                species_list=species_list,
                alat=alat,
                energy_cache=energy_cache,
                use_safe=current_use_safe,
                timeout=timeout,
            )
        except Exception as e:
            print(f"\t\tscan exception alat={alat}: {e}")
            row = None

        if row is None:
            if current_use_safe:
                safe_success_count = 0
            continue

        alats.append(row["alat"])
        energies_per_atom.append(row["energy_per_atom"])
        energy_total.append(row["energy_total"])
        ncells.append(row["ncells"])

        mode = "safe" if current_use_safe else "direct"
        print(
            f"\t\t{_species_label(species_list)} alat={row['alat']} "
            f"energy/atom={row['energy_per_atom']} mode={mode}"
        )

        if current_use_safe:
            safe_success_count += 1
            if safe_success_count >= int(safe_successes_before_direct):
                current_use_safe = False
                switched_to_direct = True
                switch_alat = row["alat"]
                print(
                    f"\t\tSwitching to direct execution after "
                    f"{safe_success_count} successful subprocess evaluations"
                )

        if early_stop_plateau:
            can_check = True
            if min_scan_alat is not None and row["alat"] < float(min_scan_alat):
                can_check = False

            if can_check and energy_plateau_detected(
                alats,
                energies_per_atom,
                window=plateau_window,
                slope_tol=plateau_slope_tol,
                range_tol=plateau_range_tol,
            ):
                plateau_stop_alat = row["alat"]
                print(f"\t\tEarly stopping: energy plateau near alat={row['alat']}")
                break

    return {
        "alats": alats,
        "energies_per_atom": energies_per_atom,
        "energy_total": energy_total,
        "ncells": ncells,
        "a_min": a_min,
        "a_max": a_max,
        "del_a": del_a,
        "initial_use_safe": bool(use_safe),
        "safe_successes_before_direct": int(safe_successes_before_direct),
        "switched_to_direct": switched_to_direct,
        "switch_alat": switch_alat,
        "early_stop_plateau": bool(early_stop_plateau),
        "plateau_stop_alat": plateau_stop_alat,
    }


def _local_minima_indices(energies_per_atom: list[float]) -> list[int]:
    from scipy.signal import find_peaks

    y = np.asarray(energies_per_atom, dtype=float)
    if len(y) == 0 or np.sum(np.isfinite(y)) == 0:
        return []

    indices, _ = find_peaks(-y)
    indices = [int(i) for i in indices if np.isfinite(y[i])]

    if len(indices) == 0:
        indices = [int(np.nanargmin(y))]

    return indices


def _starting_points_from_scan(scan: dict, max_starting_points: int) -> list[dict]:
    minima_indices = _local_minima_indices(scan["energies_per_atom"])
    minima_indices = sorted(minima_indices, key=lambda i: scan["energies_per_atom"][i])
    minima_indices = minima_indices[: int(max_starting_points)]

    return [
        {
            "index": int(i),
            "alat": float(scan["alats"][i]),
            "energy_per_atom": float(scan["energies_per_atom"][i]),
        }
        for i in minima_indices
    ]


def query_kim_influence_distance(model_name: str) -> float:
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
        return float(kim_model.get_influence_distance())
    finally:
        if hasattr(kim_model, "destroy"):
            kim_model.destroy()


def _mono_species_bounds(model: str, species: str) -> tuple[float, float, float]:
    cov = covalent_radii[atomic_numbers[species]]
    a_min = max(float(np.sqrt(2.0) * cov), 1.5)
    a_max = 12.0
    min_scan_alat = 6.5

    if not model.startswith("Sim"):
        min_cutoff = query_kim_influence_distance(model)
        if min_cutoff > a_max:
            a_max = 2.0 * min_cutoff
            min_scan_alat = min_cutoff

    return float(a_min), float(a_max), float(min_scan_alat)


def _nelder_mead_worker(
    model_name: str,
    species_list: list[str],
    start_alat: float,
    a_min: float,
    a_max: float,
    energy_bound: list[float],
    queue,
):
    """Run Nelder-Mead in a child process."""

    try:
        from scipy.optimize import minimize

        evaluations = []

        def objective(x):
            alat = float(np.ravel(x)[0])
            if not np.isfinite(alat) or alat < a_min or alat > a_max:
                return 1.0e100

            val = generate_fcc_compute_energy(
                model=model_name,
                species_list=species_list,
                alat=alat,
            )
            if val is None:
                return 1.0e100

            energy_total, ncells = val
            energy_per_atom = float(energy_total) / float(fcc_atoms_in_supercell(ncells))

            if (
                not np.isfinite(energy_per_atom)
                or abs(energy_per_atom) > energy_bound[1]
                or abs(energy_per_atom) < energy_bound[0]
            ):
                return 1.0e100

            evaluations.append(
                {
                    "alat": float(alat),
                    "energy_total": float(energy_total),
                    "energy_per_atom": float(energy_per_atom),
                    "ncells": int(ncells),
                }
            )
            return float(energy_per_atom)

        result = minimize(
            objective,
            x0=np.array([float(start_alat)]),
            method="Nelder-Mead",
            options={
                "xatol": 1.0e-4,
                "fatol": 1.0e-8,
                "maxiter": 80,
                "maxfev": 160,
                "disp": False,
            },
        )

        if len(evaluations) == 0:
            queue.put(
                {
                    "ok": False,
                    "status": "no_valid_evaluations",
                    "message": str(result.message),
                    "start_alat": float(start_alat),
                    "evaluations": evaluations,
                }
            )
            return

        best_eval = min(evaluations, key=lambda row: row["energy_per_atom"])

        if not result.success:
            queue.put(
                {
                    "ok": False,
                    "status": "optimizer_unsuccessful",
                    "message": str(result.message),
                    "start_alat": float(start_alat),
                    "best_alat": float(best_eval["alat"]),
                    "best_energy_per_atom": float(best_eval["energy_per_atom"]),
                    "evaluations": evaluations,
                }
            )
            return

        queue.put(
            {
                "ok": True,
                "status": "success",
                "message": str(result.message),
                "start_alat": float(start_alat),
                "good_alat": float(best_eval["alat"]),
                "good_energy_total": float(best_eval["energy_total"]),
                "good_energy_per_atom": float(best_eval["energy_per_atom"]),
                "good_ncells": int(best_eval["ncells"]),
                "optimizer_x": float(np.ravel(result.x)[0]),
                "optimizer_fun": float(result.fun),
                "nfev": int(result.nfev),
                "nit": int(result.nit),
                "evaluations": evaluations,
            }
        )

    except Exception:
        queue.put(
            {
                "ok": False,
                "status": "python_exception",
                "message": traceback.format_exc(),
                "start_alat": float(start_alat),
                "evaluations": [],
            }
        )


def scipy_nelder_mead_safe(
    model: str,
    species_list: list[str],
    start_alat: float,
    a_min: float,
    a_max: float,
    energy_bound: list[float],
    timeout: float = 120.0,
) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_nelder_mead_worker,
        args=(model, species_list, start_alat, a_min, a_max, energy_bound, queue),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "ok": False,
            "status": "timeout",
            "message": f"Nelder-Mead timed out after {timeout} seconds",
            "start_alat": float(start_alat),
            "evaluations": [],
        }

    if proc.exitcode != 0:
        if proc.exitcode < 0:
            sig = -proc.exitcode
            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = f"signal {sig}"
            message = f"child died with {sig_name}"
        else:
            message = f"child exit code {proc.exitcode}"
        return {
            "ok": False,
            "status": "crash",
            "message": message,
            "start_alat": float(start_alat),
            "evaluations": [],
        }

    if queue.empty():
        return {
            "ok": False,
            "status": "no_result",
            "message": "child exited but returned no result",
            "start_alat": float(start_alat),
            "evaluations": [],
        }

    return queue.get()


def _failure_config_result(
    species_list: list[str],
    configuration_type: str,
    scan: dict | None,
    starting_points: list[dict],
    attempts: list[dict],
    reason: str,
    bounds: dict,
) -> dict:
    return {
        "ok": False,
        "status": "failed",
        "failure_reason": reason,
        "configuration_type": configuration_type,
        "species_list": species_list,
        "species_label": _species_label(species_list),
        "good_alat": -1.0,
        "good_energy_per_atom": None,
        "good_energy_total": None,
        "good_ncells": -1,
        "bounds": bounds,
        "coarse_scan": scan,
        "coarse_minima": starting_points,
        "nelder_mead_attempts": attempts,
        "search_strategy": "coarse_scan_plus_nelder_mead_no_led",
        "all_alats": [],
        "all_energies_per_atom": [],
    }


def _equilibrate_one_config(
    model: str,
    species_list: list[str],
    a_min: float,
    a_max: float,
    min_scan_alat: float,
    configuration_type: str,
    energy_bound: list[float],
    coarse_del_a: float,
    safe_successes_before_direct: int,
    coarse_timeout: float,
    nelder_mead_timeout: float,
    max_starting_points: int,
) -> dict:
    print(f"\n\tCONFIG: {_species_label(species_list)} ({configuration_type})")
    print(f"\tSCAN RANGE: [{a_min}, {a_max}], del_a={coarse_del_a}")

    energy_cache = {}
    use_safe = True
    scan = _scan_alat_range(
        model=model,
        species_list=species_list,
        a_min=a_min,
        a_max=a_max,
        del_a=coarse_del_a,
        energy_cache=energy_cache,
        use_safe=use_safe,
        safe_successes_before_direct=safe_successes_before_direct,
        timeout=coarse_timeout,
        early_stop_plateau=True,
        min_scan_alat=min_scan_alat,
    )

    bounds = {
        "a_min": float(a_min),
        "a_max": float(a_max),
        "min_scan_alat_for_plateau": float(min_scan_alat),
        "coarse_del_a": float(coarse_del_a),
    }

    if len(scan["alats"]) == 0:
        return _failure_config_result(
            species_list,
            configuration_type,
            scan,
            [],
            [],
            "coarse_scan_has_no_successful_points",
            bounds,
        )

    starting_points = _starting_points_from_scan(
        scan,
        max_starting_points=max_starting_points,
    )

    if len(starting_points) == 0:
        return _failure_config_result(
            species_list,
            configuration_type,
            scan,
            [],
            [],
            "no_coarse_minima_found",
            bounds,
        )

    attempts = []
    for start in starting_points:
        print(f"\tNelder-Mead start alat={start['alat']}")
        attempt = scipy_nelder_mead_safe(
            model=model,
            species_list=species_list,
            start_alat=start["alat"],
            a_min=a_min,
            a_max=a_max,
            energy_bound=energy_bound,
            timeout=nelder_mead_timeout,
        )
        attempts.append(attempt)
        if attempt.get("ok", False):
            evaluations = attempt.get("evaluations", [])
            return {
                "ok": True,
                "status": "success",
                "configuration_type": configuration_type,
                "species_list": species_list,
                "species_label": _species_label(species_list),
                "good_alat": float(attempt["good_alat"]),
                "good_energy_per_atom": float(attempt["good_energy_per_atom"]),
                "good_energy_total": float(attempt["good_energy_total"]),
                "good_ncells": int(attempt["good_ncells"]),
                "bounds": bounds,
                "coarse_scan": scan,
                "coarse_minima": starting_points,
                "nelder_mead_attempts": attempts,
                "selected_attempt_index": len(attempts) - 1,
                "search_strategy": "coarse_scan_plus_nelder_mead_no_led",
                "all_alats": [float(row["alat"]) for row in evaluations],
                "all_energies_per_atom": [
                    float(row["energy_per_atom"]) for row in evaluations
                ],
            }

    return _failure_config_result(
        species_list,
        configuration_type,
        scan,
        starting_points,
        attempts,
        "all_nelder_mead_attempts_failed",
        bounds,
    )


def _finite_positive_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    if len(arr) == 0:
        return -1.0
    return float(np.mean(arr))


def find_equilibrium_config_FCC(
    model: str,
    species_list: list[str],
    energy_bound: list[float] = [5.0e-2, 5.0e2],
    coarse_del_a: float = 0.1,
    mixed_coarse_del_a: float = 0.1,
    safe_successes_before_direct: int = 10,
    coarse_timeout: float = 300.0,
    nelder_mead_timeout: float = 120.0,
    max_starting_points: int = 6,
) -> dict:
    """
    Find an FCC equilibrium configuration using only:
        coarse lattice sweep -> local minima -> Nelder-Mead refinement.

    model is always expected to be a KIM model name string.
    species_list is always expected to be list[str]. 

    For each individual species, this first computes a mono-species FCC
    equilibrium using species_list=[species]. If the input species_list has only
    one element, that mono-species result is the final result.

    If the input species_list has more than one element, the average of the
    successful mono-species equilibrium lattice constants is used as a good
    guess for the mixed-species FCC. The mixed-species FCC is then swept over
    [0.75*good_guess, 2.0*good_guess], and Nelder-Mead is run from the coarse
    mixed-species minima.
    """

    mono_results = []
    for species in species_list:
        mono_species_list = [species]
        a_min, a_max, min_scan_alat = _mono_species_bounds(model, species)
        mono = _equilibrate_one_config(
            model=model,
            species_list=mono_species_list,
            a_min=a_min,
            a_max=a_max,
            min_scan_alat=min_scan_alat,
            configuration_type="mono_species_fcc",
            energy_bound=energy_bound,
            coarse_del_a=coarse_del_a,
            safe_successes_before_direct=safe_successes_before_direct,
            coarse_timeout=coarse_timeout,
            nelder_mead_timeout=nelder_mead_timeout,
            max_starting_points=max_starting_points,
        )
        mono_results.append(mono)

    mono_good_alats = [row.get("good_alat", -1.0) for row in mono_results]
    approx_mixed_equilibrium_alat = _finite_positive_mean(mono_good_alats)

    if len(species_list) == 1:
        mixed_result = None
        final_result = mono_results[0]
        results = mono_results
    else:
        if approx_mixed_equilibrium_alat <= 0.0:
            mixed_result = {
                "ok": False,
                "status": "failed",
                "configuration_type": "mixed_species_fcc",
                "species_list": species_list,
                "species_label": _species_label(species_list),
                "good_alat": -1.0,
                "good_energy_per_atom": None,
                "good_energy_total": None,
                "good_ncells": -1,
                "failure_reason": "no_successful_mono_species_equilibria_for_average_guess",
                "search_strategy": "coarse_scan_plus_nelder_mead_no_led",
                "approx_mixed_equilibrium_alat": approx_mixed_equilibrium_alat,
            }
        else:
            mixed_a_min = 0.75 * approx_mixed_equilibrium_alat
            mixed_a_max = 2.0 * approx_mixed_equilibrium_alat
            mixed_min_scan_alat = approx_mixed_equilibrium_alat
            mixed_result = _equilibrate_one_config(
                model=model,
                species_list=species_list,
                a_min=mixed_a_min,
                a_max=mixed_a_max,
                min_scan_alat=mixed_min_scan_alat,
                configuration_type="mixed_species_fcc",
                energy_bound=energy_bound,
                coarse_del_a=mixed_coarse_del_a,
                safe_successes_before_direct=safe_successes_before_direct,
                coarse_timeout=coarse_timeout,
                nelder_mead_timeout=nelder_mead_timeout,
                max_starting_points=max_starting_points,
            )
            mixed_result["approx_mixed_equilibrium_alat"] = approx_mixed_equilibrium_alat
            mixed_result["mixed_sweep_rule"] = "[0.75 * average_mono_equilibrium, 2.0 * average_mono_equilibrium]"

        final_result = mixed_result
        results = mono_results + [mixed_result]

    return {
        "model": model,
        "species_list": species_list,
        "species_label": _species_label(species_list),
        "ncells_per_side": FWC_NCELLS_PER_SIDE,
        "method": "coarse_scan_plus_nelder_mead_no_led",
        "mono_species_equilibrium_alats": {
            row["species_label"]: row.get("good_alat", -1.0) for row in mono_results
        },
        "approx_mixed_equilibrium_alat": approx_mixed_equilibrium_alat,
        "mono_species_results": mono_results,
        "mixed_species_result": mixed_result,
        "final_result": final_result,
        "equilibrium_alat": final_result.get("good_alat", -1.0) if final_result else -1.0,
        "results": results,
    }
