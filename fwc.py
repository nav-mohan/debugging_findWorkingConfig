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
# this is only for monospecies
# species will be turned into a string, not a list
def generate_fcc_compute_energy(
    model: Union[str, Calculator],
    species: list,
    alat: float,
    seed: Union[int, None] = 13,
) -> tuple[float, int]:
    """
    Construct an FCC lattice large enough to accommodate all species,
    evaluate its energy, and return the energy and size of the supercell.

    Args:
        model: The model name or calculator object.
        species: List of atomic species to be incorporated into the FCC lattice.
        alat: The lattice constant for the FCC lattice.
        seed: Optional random seed for reproducibility during species randomization.

    Returns:
        A tuple containing:
            - The total potential energy of the lattice.
            - The number of unit cells along each side of the supercell.
    """

    ncells_per_side = 1
    while True:
        atoms = FaceCenteredCubic(
            size=(ncells_per_side, ncells_per_side, ncells_per_side),
            latticeconstant=alat,
            symbol="H",
            # pbc=False,
            pbc=True,
        )
        if len(atoms) < len(species):
            ncells_per_side += 1
        else:
            break
    random.seed(seed)
    randomize_species(atoms, species)
    if isinstance(model, str):
        calc = KIM(model)
    else:
        calc = model
    atoms.set_calculator(calc)

    # compute energy
    try:
        pe = atoms.get_potential_energy()
        # General clean-up
        if hasattr(calc, "clean"):
            calc.clean()
        if hasattr(calc, "__del__"):
            calc.__del__()
        return pe, ncells_per_side
    except Exception as e:
        raise (e)


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
def local_edge_detection(x: list, y: list) -> list:
    """
    Computes the Local Edge Detection (LED) values for the x-y curve.
    The algorithm helps to identify discontinuities or abrupt changes in the curve.

    Based on the paper:
    A. Gelb and E. Tadmor, "Local edge detection for non-linear signals,"
    Journal of Scientific Computing, 28:279-306, 2006.

    Args:
        x: A list of x-values representing the independent variable of the curve.
        y: A list of y-values representing the dependent variable of the curve.

    Returns:
        A list of LED values of strength of discontinuities in the x-y curve.

    Notes:
        The LED[i] corresponds to the X[i+2]

    """

    led_values = []

    fact = 1.0 / 6.0
    for j in range(2, len(y) - 3):
        # use 5-th order local difference formula
        led = fact * (
            -y[j - 2]
            + 5 * y[j - 1]
            - 10 * y[j]
            + 10 * y[j + 1]
            - 5 * y[j + 2]
            + y[j + 3]
        )
        led_values.append(led)

    return led_values


################################################################################
def filter_good_alat(
    alats: list,
    energies: list,
    ncells: list,
    leds: list,
    min_cutoff: float,
    etol: list = [5e-2, 5e2],
    led_tol: float = 1.0,
) -> dict:
    """
    Filter a good lattice constant (alat) based on the given criteria.

    This function filters out a valid alat value from the provided lists of alats,
    energies, number of cells, and LED values. The filtering is done based on several
    conditions including the energy bounds, LED tolerance, and minimum cutoff distance.

    Args:
        alats: A list of lattice parameters (alat).
        energies: A list of energy-per-atom values corresponding to each alat.
                          Used to compute the total energy of the FCC structure
                          for filtering.
        ncells: A list of the number of cells corresponding to each alat.
                        Used to compute the number of atoms for filtering by energy.
        leds: A list of LED values corresponding to each alat.
                     Note that leds[i] corresponds to alats[i+2]
        min_cutoff: The minimum cutoff distance for the model-species
                            combination. This is used to filter alats.
        etol: A list containing the minimum and maximum energy-per-atom
                               bounds for filtering. Default is [5e-2, 5e2].
        led_tol: The maximum LED value for filtering. Default is 1.0.

    Returns:
        dict: A dictionary containing the filtered alat value and related properties:
            - 'good_alat': The filtered alat value.
            - 'min_led': The minimum LED value found.
            - 'good_ncells': The number of cells corresponding to the selected alat.

    Notes:
        The following conditions are applied to filter the valid alat:
        - `0.2 * min_cutoff < alat < 0.8 * min_cutoff`
        - `etol[0] < |energy-per-atom| < etol[1]`
        - `|LED| < led_tol`
        - `0 < |LED| < min_led`

    """

    min_led = np.inf
    good_alat = None
    good_ncells: int = 0

    N = len(alats)
    valid_ncell = []
    valid_energy = []
    valid_leds = []
    valid_alats = []

    for i in range(2, N - 3):
        alat = alats[i]
        energy = energies[i]
        ncell = ncells[i]
        led = leds[i - 2]  # leds[0] corresponds to alats[2]
        natoms = fcc_atoms_in_supercell(ncell)

        # r_nn = alat / np.sqrt(2)
        # if min_cutoff is not None and r_nn > 1.8 * min_cutoff:
            # print(f"r_nn = {r_nn} SKIPPING 1.2")
            # continue
        if abs(energy) > etol[1] * natoms or abs(energy) < etol[0] * natoms:
            # print(f"r_nn = {r_nn} SKIPPING ENERGY")
            continue
        # if abs(led) > led_tol or abs(led) > min_led:
        if abs(led) > led_tol:
            # print(f"r_nn = {r_nn} SKIPPING LED {led}")
            continue
        valid_leds.append(led)
        valid_ncell.append(ncell)
        valid_energy.append(energy)
        valid_alats.append(alat)
        # good_alat = alat
        # good_ncells = ncell
        # min_led = abs(led)


    if len(valid_alats) == 0:
        return {
            "good_alat": -1.0, 
            "min_led": -1.0, 
            "good_ncells": -1, 
            "valid_alats":[-1.0], 
            "valid_energy":[0], 
            "indices":[0],
            'min_index' : 0
            }

    valid_leds = np.array(valid_leds).tolist()
    valid_ncell = np.array(valid_ncell).tolist()
    valid_energy = np.array(valid_energy).tolist()
    valid_alats = np.array(valid_alats).tolist()
    
    # now pick the local minima energy from the valid_energy
    from scipy.signal import find_peaks

    # Negate array to turn minima into peaks
    indices, _ = find_peaks(-np.array(valid_energy))
    indices = indices.tolist()
    
    # if there are more than one peak, pick the lowest energy 
    if len(indices) == 0:
        min_index = int(np.argmin(valid_energy))
    else:
        min_index = min(indices, key=lambda i: valid_energy[i])

    good_alat = valid_alats[min_index] 
    min_led = valid_leds[min_index] 
    good_ncells = valid_ncell[min_index] 


    return {
        "good_alat": good_alat, 
        "min_led": min_led, 
        "good_ncells": good_ncells, 
        "valid_alats":valid_alats, 
        "valid_energy":valid_energy, 
        "indices":indices,
        'min_index' : min_index
        }


################################################################################

# this is only for mono-species. 
# species will be turned into a string, not a list 
def find_working_configuration_FCC(
    model: Union[str, Calculator],
    species: list,
    energy_bound: list = [5e-2, 5e2],
    led_tol: float = 1.0,
    seed: Union[int, None] = 13,
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
        seed: An optional random seed for reproducibility.

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

    min_cutoff = None
    # try:
    #     min_cutoff = get_model_species_minimum_cutoff(model, species)
    #     amin = np.sqrt(2) * 0.3 * min_cutoff
    #     amax = np.sqrt(2) * 1.0 * min_cutoff
    # except Exception:
    #     min_cutoff = None
    #     amin = 0.5
    #     amax = 8.0

    # this amin is set to prevent segfaults . For example large Plutonium atom in a SIM_LAMMPS models throws a segfault that cannot be recovered from. 
    # but if energy-minima is < 2.5 then we will miss that. in that case what will likely happen is that we will exhaust the search range and the energy minima will be somewhere near amax
    # if we exhaust this search-range then redo the search-range but with amin = 1.5, amax = 3.0

    from ase.data import atomic_numbers, covalent_radii
    # species is just a list of length 1 
    cov = covalent_radii[atomic_numbers[species[0]]]
    amin = max(np.sqrt(2) * cov , 1.5) # strict lower bound of 1.5 to avoid high-density
    # amin = 2.5 
    amax = 12.0
    del_a = 0.01
    na = int(math.ceil((amax - amin) / del_a))

    alats = []
    energies = []
    ncells = []  # will be used for filtering by energy-per-atom

    for j in range(0, na + 1):
        a = amin + j * del_a
        try:
            val = generate_fcc_compute_energy(model, species, a, seed)
            if val is not None:
                alats.append(a)
                energies.append(val[0])  # first value is energy
                ncells.append(val[1])  # second value is number of atoms

        except Exception:
            continue

    # these LED values correspond to alats[2] ... alats[na-3]
    leds = local_edge_detection(alats, energies)
    return filter_good_alat(
        alats, energies, ncells, leds, min_cutoff, energy_bound, led_tol
    )

