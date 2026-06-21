"""
Example:
python3 script_eq_config_fcc.py \
  --input-module input_data_REAXFFBUCK \
  --output-dir plotdata_equilibrium_config_fcc \
  --nelder-mead-timeout 120 \
  --max-starting-points 6
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from find_eq_config_fcc import find_equilibrium_config_FCC

import os 
DEVICE = os.getenv("KIM_MODEL_EXECUTION_DEVICE","cpu")
def output_filename(output_dir: Path, model: str, species_list: list[str]) -> Path:
    species_label = "-".join(species_list)
    return output_dir / f"{model}_{species_label}_{DEVICE}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-module", required=True)
    parser.add_argument("--output-dir", default="plotdata_equilibrium_config_fcc")
    parser.add_argument("--coarse-del-a", type=float, default=0.1)
    parser.add_argument("--mixed-coarse-del-a", type=float, default=0.1)
    parser.add_argument("--safe-successes-before-direct", type=int, default=10)
    parser.add_argument("--coarse-timeout", type=float, default=300.0)
    parser.add_argument("--nelder-mead-timeout", type=float, default=120.0)
    parser.add_argument("--max-starting-points", type=int, default=6)
    args = parser.parse_args()

    input_module = importlib.import_module(args.input_module)
    input_data = input_module.input_data

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in input_data:
        model = entry["model"]
        species_list = entry["species"]
        model_shortname = entry.get("model_shortname", model)

        print("\n#######################################################")
        print(model_shortname)
        print(model)
        print("species_list:", species_list)

        result = find_equilibrium_config_FCC(
            model=model,
            species_list=species_list,
            coarse_del_a=args.coarse_del_a,
            mixed_coarse_del_a=args.mixed_coarse_del_a,
            safe_successes_before_direct=args.safe_successes_before_direct,
            coarse_timeout=args.coarse_timeout,
            nelder_mead_timeout=args.nelder_mead_timeout,
            max_starting_points=args.max_starting_points,
        )

        result["model_shortname"] = model_shortname

        out_path = output_filename(output_dir, model, species_list)
        with out_path.open("w") as f:
            json.dump(result, f, indent=4)

        print("WROTE:", out_path)
        print("equilibrium_alat:", result.get("equilibrium_alat"))
        final_result = result.get("final_result") or {}
        print("final_status:", final_result.get("status"))
        print("final_configuration_type:", final_result.get("configuration_type"))


if __name__ == "__main__":
    main()
