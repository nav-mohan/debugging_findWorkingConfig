# i initially wrote this assuming mixed-species 
# but later i changed it to mono-species
# so i should rename functions and change-function-signatures to reflect that 
# rename to "find_working_configuration_monospeciesFCC"
# create a wrapper "find_working_configuration_mixedspeciesFCC" that iteratively calls "find_working_configuration_monospeciesFCC"


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


# this is only for mono-species. 
# species will be turned into a string, not a list 
def find_working_configuration_FCC(
    model: Union[str, Calculator],
    species: str,
    energy_bound: list = [5e-2, 5e2],
    led_tol: float = 1.0,
    led_order: int = 5,
) -> dict:

    """
        Find an FCC configuration for a model and species with energy and LED constraints.

        This function constructs an FCC configuration for the specified species and model,
        and filters the configurations based on energy-per-atom and smoothness of the
        energy-alat relation. The energy values must lie within the specified bounds,
        and the local edge detection (LED) must be below the given tolerance.

        Args:
            model: The model name or calculator object used to
                                            compute the energies of the configurations.
            species: atomic species to be incorporated into the FCC structure.
            energy_bound: A list of two values specifying the minimum
                                        and maximum energy-per-atom bounds for filtering.
                                        Default is [5e-2, 5e2].
            led_tol: The tolerance for local edge detection filtering.
                                    Default is 1.0.
            led_order: LED stencil order. Supported values are 3, 5, 7, and 9.
                                    Default is 5.

        Returns:
            dict: A dictionary containing the working FCC configuration, including the
                optimal lattice constant, number of cells per side, and the
                corresponding LED value.

        Notes:
            - The function first computes the minimum cutoff distance for the species pair.
            - It then attempts different lattice constants and filters the configurations
            based on energy-per-atom and LED values.
            - Only configurations that meet the energy and LED criteria are retained.
    """

    _validate_led_order(led_order)

    from ase.data import atomic_numbers, covalent_radii
    # species is just a list of length 1 
    cov = covalent_radii[atomic_numbers[species]]
    # print("got covalent radius ", species[0], cov)

    amin = max(np.sqrt(2) * cov , 1.5) # strict lower bound of 1.5 to avoid high-density
    amax = 12.0
    min_scan_alat = 6.5 # atleast scan until alat=6.5A before considering early termination
    plateau_detection_window = 20 # if the energy has been a platue over the last 0.2A
    plateau_detection_slope = 1e-3 # if slope(energy-vs-alat) within the window is lower than this tol, that is a plateau
    plateau_detection_range = 5e-4 # if abs(E_max - E_min) within the window lie within this tol, that is a plateau 
    ran_without_exceptions = 25 # if the last 0.25A evaluations have ran without exceptions then we can stop spawning subprocesses and directly call generate_fcc_compute_energy
    if "LAMMPS" not in model: 
        min_cutoff = query_kim_influence_distance(model)
        if min_cutoff > amax: 
            amax = 2*min_cutoff
            min_scan_alat = min_cutoff

    del_a = 0.01

    # DEBUGGIN the 2 SIM_AIREBO models 
    # amin,amax,del_a = 2.0,3.5,0.001 # DEBUGGING!!! 

    na = int(math.ceil((amax - amin) / del_a))
    # print(f"amin,amax {amin},{amax}")
    alats = []
    energies_per_atom = []
    ncells = []  # will be used for computing LED of energy-per-atom

    for j in range(0, na + 1):
        a = amin + j * del_a
        try:
            val = None
            if ran_without_exceptions >= 0:
                val = generate_fcc_compute_energy_safe(model, species, a)
            else:
                val = generate_fcc_compute_energy(model, species, a)
            if val is not None:
                alats.append(a)
                # val[0] is total energy of FCC, val[1] is number-of-cells in FCC 
                energy_total = val[0]
                ncell = val[1]
                natoms = fcc_atoms_in_supercell(ncell)
                energy_pa = energy_total/natoms
                energies_per_atom.append(energy_pa)
                ncells.append(ncell)
                print(f"\t\talat = {a} | energy = {val[0]}")

                if a > min_scan_alat and energy_plateau_detected(alats,energies_per_atom,window=plateau_detection_window,slope_tol=plateau_detection_slope,range_tol=plateau_detection_range,):
                    print(f"\t\tEarly stopping: energy plateau detected near alat = {a}")
                    break

                ran_without_exceptions -= 1
                print(f"\t\tDecrement counter alat = {a} | counter = {ran_without_exceptions}")

        except Exception as e:
            print("gen-fcc exception",e)
            continue
    # LED[k] corresponds to alats[k + (led_order - 1)//2].
    leds = local_edge_detection(alats, energies_per_atom, order=led_order)
    return filter_good_alat(
        alats,
        energies_per_atom,
        leds,
        energy_bound,
        led_tol,
        led_order=led_order,
    )


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

