import json 

import numdifftools as nd
import numpy as np
import math

import kim_tools.ase as kim_ase_utils
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.kim import KIM, get_model_supported_species
import scipy.optimize

from fwc import *

######################################################################################################
# choose which input_data to import, whether PORTABLE-MODEL, SIM_MODEL or TORCH_MODEL
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--input-module", required=True)
args = parser.parse_args()

input_module = importlib.import_module(args.input_module)
input_data = input_module.input_data

# to choose which type of models to run, execute script.py like so
# python script.py --input-module input_data_TORCH
# python script.py --input-module input_data_PORT
# python script.py --input-module input_data_SIM
######################################################################################################

def cubic_cell_energy(alat, atoms, ncells_per_side):
    """
    Calculate the energy of the passed 'atoms' structure containing a
    cubic structure with 'ncells_per_side'. Scale to lattice constant
    'alat' (passed as a nd array of length 1) and return the energy.
    """
    acell = alat[0] * ncells_per_side
    atoms.set_cell([acell, acell, acell], scale_atoms=True)
    e = atoms.get_potential_energy()
    return e

def find_equilibrium_fcc(model,species, ncells_per_side = 1, grid_stepsize=0.01, min_alat = 2.5, max_alat = 10.0):
    alat_ave = []
    for spec in species:
        # Check if this species has non-trivial force and energy interactions
        atoms_interacting_energy, atoms_interacting_force = kim_ase_utils.check_if_atoms_interacting(
            model, symbols=[spec, spec]
        )
        if not atoms_interacting_energy:
            print("")
            print(
                "WARNING: The model provided, {}, does not possess a non-trivial energy "
                "interaction for species {} as required by this Verification "
                "Check. Skipping...".format(model, spec)
            )
            print("")
            continue

        if not atoms_interacting_force:
            print("")
            print(
                "WARNING: The model provided, {}, does not possess a non-trivial force "
                "interaction for species {} as required by this Verification Check.  "
                "Skipping...".format(model, spec)
            )
            print("")
            continue

        # find equilibrium lattice constant, so that the numerical derivatives
        # of all potentials are evaluated in a similar portion of their
        # potential energy surface, making comparisons between potentials
        # more meaningful.
        calc = KIM(model)
        alat = min_alat
        done = False
        while not done:
            atoms = FaceCenteredCubic(
                size=(ncells_per_side,ncells_per_side,ncells_per_side), 
                latticeconstant=alat, 
                symbol=spec, 
                # pbc=False
                pbc=True
            )
            atoms.set_calculator(calc)
            try:
                res = scipy.optimize.minimize(
                    cubic_cell_energy,
                    alat,
                    args=(atoms, ncells_per_side),
                    method="Nelder-Mead",
                    tol=1e-6,
                )
                alat = res.x[0]
                done = True
            except:  # noqa: E722
                # failed for some reason (assume it's because of KIM error)
                alat += grid_stepsize
                if alat > max_alat:
                    done = True
        alat_ave.append(alat)

    if len(alat_ave) == 0:
        alat_ave = [np.float64(-1.0)]
    return np.mean(alat_ave)


for data in input_data:
    model,species,model_shortname = data["model"], data["species"], data["model_shortname"]
    print(model_shortname)
    data = []
    results = []
    avg_equil_alat = []
    avg_good_alat = []
    for spec in species:
        print(f"\tSPEC: {spec}")
        spec_equil_alat = -1.0
        spec_equil_alat = find_equilibrium_fcc(model,[spec]) # comment out for ML models
        spec_work_config = find_working_configuration_FCC(model=model,species=[spec])

        spec_good_alat = spec_work_config['good_alat'] # use this for calculating difference between spec_equil_alat
        spec_valid_alats = spec_work_config['valid_alats']
        spec_valid_energy = spec_work_config['valid_energy'] 
        spec_min_indices = spec_work_config['indices'] # highlight these in green
        spec_min_index = spec_work_config['min_index'] # highlight this in red

        if len(spec_valid_alats) == 0:
            print("FAILED TO FIND A SINGLE VALID",model)
            with open("notes.txt", "a") as file:
                file.write(f"\t{model} {spec_equil_alat}\n")


        results.append(
            {
                "spec" : spec, # make subplot for each spec
                "spec_equil_alat" : spec_equil_alat, # draw a vertical line in each subplot
                "spec_good_alat" : spec_good_alat, # calculate relative-difference from spec_equil_alat in each subplot
                "spec_valid_alats" : spec_valid_alats, # x-axis for each subplot
                "spec_valid_energy" : spec_valid_energy, # y-axis for each subplot
                "spec_min_indices" : spec_min_indices, # color these green in each subplot
                "spec_min_index" : spec_min_index, # color this red in each subplot
            }
        )

        avg_equil_alat.append(spec_equil_alat)
        avg_good_alat.append(spec_good_alat)

        print(f"\t\tEQUIL: {float(spec_equil_alat)}")
        print(f"\t\tGOOD : {float(spec_good_alat)}")
        print(f"\t\tRELD : {float(100*(spec_good_alat - spec_equil_alat)/(spec_equil_alat))}")


    data.append(
        {
            "avg_equil_alat" : np.mean(avg_equil_alat).item(),
            "avg_good_alat" : np.mean(avg_good_alat).item(),
            # calculate relative-difference between these two and put it in title of graph
            "results" : results
        }
    )
    print(f"\tAVG_EQUIL: {float(np.mean(avg_equil_alat).item())}")
    print(f"\tAVG_GOOD : {float(np.mean(avg_good_alat).item())}")
    
    print("\n#######################################################\n")

    # write data to file {model}.json
    # make subplots for each spec
    with open(f"plotdata/{model}_{'-'.join(species)}.json", "w") as file:
        json.dump(data, file, indent=4)


