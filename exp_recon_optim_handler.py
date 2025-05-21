import argparse
import json
from experiment.scenario import Scenario, Experiment
from experiment_setup.exp_utrecht_10_492_594_v2_setup import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script takes a single scenario number "
                                                 "for which to optimize the reconstruction parameters.")
    parser.add_argument("--scenario_id", type=int, help="Scenario number")
    parser.add_argument("--continue_previous", action="store_true", help="Option to continue aborted run")
    args = parser.parse_args()

    print("\n=========================================================================================================")
    print("Starting exp_recon_optim_crasher.py to optimize_reconstruction_params() or crash safely.")

    # ==================================================================================================================
    # Set up or load the experiment instance and print details about the settings

    load_existing_experiment = True  # set false to apply changes made to the config

    if not load_existing_experiment:
        e = Experiment(experiment_name, experiment_dirpath, default_config, scenario_settings, scene_parts)

        print("\nSetting up experiment ...")
        e.setup()
    else:
        print("\nLoading existing experiment ...")
        e = Experiment.load(experiment_dirpath / experiment_name, load_scenarios=True)

    # print("\nScenario settings:\n")
    #
    # for i, ss in enumerate(scenario_settings):
    #     print(str(i) + ": " + str(ss))
    #
    # print("\nTarget densities and point spacing:\n")
    # print(json.dumps(dict(zip(target_densities, point_spacings)), indent=4))
    #
    # print("\nUnique surveys:")
    # print(scenarios_unique_surveys)

    # ==================================================================================================================
    # Prepare and perform the optimization step

    print(f"\nArguments received:")
    print(f"- Scenario number: {args.scenario_id}")
    print(f"- Continue: {args.continue_previous}")

    geoflow_default_params_for_optim = {
        k: glb.geoflow_parameters_default[k]
        for k in [k2.split("_", 1)[1] for k2 in glb.geoflow_optim_parameter_space_narrow_2.keys()]
    }

    e.optimize_reconstruction_params(
        sequence=[args.scenario_id],
        init_points=5,
        n_iter=45,
        recon_timeout=600,
        probe_parameter_sets=[geoflow_default_params_for_optim],
        continue_from_previous_run=args.continue_previous
    )
