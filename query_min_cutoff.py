from ase.data import atomic_numbers, covalent_radii
import kimpy
import argparse
import importlib



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


parser = argparse.ArgumentParser()
parser.add_argument("--input-module", required=True)
args = parser.parse_args()

input_module = importlib.import_module(args.input_module)
input_data = input_module.input_data

for data in input_data:
    model,species,model_shortname = data["model"], data["species"], data["model_shortname"]
    print(model_shortname)
    min_cutoff = query_kim_influence_distance(model)
    print("min_cutoff = ",min_cutoff)
    for spec in species:
        cov = covalent_radii[atomic_numbers[spec]]
        print(f"\t{spec} COV ", cov)
        # amin = max(1.414 * cov , 1.5) # strict lower bound of 1.5 to avoid high-density
        # amax = 12.0
        # if min_cutoff > amax: 
            # amax = 2*min_cutoff
