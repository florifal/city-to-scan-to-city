import json
import shutil
from datetime import datetime

from experiment.scenario import Scenario, Experiment
from experiment_setup.exp_utrecht_10_492_594_v2_setup import *
from experiment.utils import execute_subprocess, get_most_recently_created_folder
from experiment.reconstruction import ReconstructionError

if __name__ == "__main__":
    load_existing_experiment = False  # set false to apply changes made to the config

    if not load_existing_experiment:
        e = Experiment(experiment_name, experiment_dirpath, default_config, scenario_settings, scene_parts)

        print("\nSetting up experiment ...")
        e.setup()
    else:
        print("\nLoading existing experiment ...")
        e = Experiment.load(experiment_dirpath / experiment_name, load_scenarios=True)

    print("\nScenario settings:\n")

    for i, ss in enumerate(scenario_settings):
        print(str(i) + ": " + str(ss))

    print("\nTarget densities and point spacing:\n")
    print(json.dumps(dict(zip(target_densities, point_spacings)), indent=4))

    print("\nUnique surveys:")
    print(scenarios_unique_surveys)

    # ==================================================================================================================
    # Run surveys by calling a separate script to handle the crashes that PyHelios recently started causing at the end
    # of each survey simulation

    # s_ids = range(4, 12)
    #
    # for s_id in s_ids:
    #     command = [
    #         "python",
    #         Path(r"C:\Users\Florian\OneDrive - TUM\Universität\24\Master's Thesis\Code\city-to-scan-to-city\exp_survey_crasher.py"),
    #         str(s_id)
    #     ]
    #
    #     print(f"\n[{t}] Running experiment.py for ID {s_id} ...")
    #     print(f"Command:\n{command}")
    #     # t = datetime.today().strftime("%y%m%d-%H%M%S")
    #     # log_filename = f"survey_{s_id}_at_{t}.log"
    #
    #     try:
    #         for line in execute_subprocess(command):
    #             pass
    #         # The log file has around 50 MB, so I decided against keeping it
    #         # with open(log_filename, "w", encoding="utf-8") as f:
    #         #     for line in execute_subprocess(command):
    #         #         f.write(line)
    #     except Exception as e:
    #         print("\nError occurred as expected:")
    #         print(str(e))
    #     else:
    #         print("\nNo error occurred unexpectedly.")
    #
    #     print("Continuing with next ID.\n")

    # ==================================================================================================================
    # Merge point cloud swaths independently after running the crashing survey with exp_survey_wrapper.py

    # A) Run complete merging procedure, filtering included

    # e.run_steps([
    #     Scenario.setup_survey,
    #     Scenario.prepare_survey,
    #     Scenario.merge_clouds,
    #     Scenario.clear_survey
    # ],
    #     scenarios=scenarios_unique_surveys
    # )

    # B) Only run the merging part if clouds were filtered previously

    # for scenario_name in scenarios_unique_surveys:
    #     e.scenarios[scenario_name].setup_survey()
    #     e.scenarios[scenario_name].prepare_survey()
    #     e.scenarios[scenario_name].survey.setup_merger()
    #     e.scenarios[scenario_name].survey.cloud_merger.merge_clouds()
    #     e.scenarios[scenario_name].survey.cloud_merger.clear()

    # ==================================================================================================================
    # For those scenarios that only add noise to a point cloud of a noise-free scenario (for which the unique surveys
    # were run), copy the file `merged_cloud_filepath.txt` to the survey directory, which points to the point cloud of
    # the corresponding noise-free unique survey.
    # To be able to copy the file, first create the scenario's survey directories by setting up and preparing the survey

    # e.run_steps([
    #     Scenario.setup_survey,
    #     Scenario.prepare_survey,
    #     Scenario.clear_survey
    # ])
    #
    # n_density_levels = len(target_densities)
    # n_noise_levels = len(error_steps)
    #
    # for i in [j * n_noise_levels for j in range(n_density_levels)]:
    #     print(i)
    #     merged_cloud_textfile_filepath = e.scenarios[f"scenario_{i:03}"].textfile_merged_cloud_path_filepath
    #     for k in range(1, n_noise_levels):
    #         print(f" {i + k}")
    #         shutil.copy2(
    #             merged_cloud_textfile_filepath,
    #             e.scenarios[f"scenario_{(i + k):03}"].textfile_merged_cloud_path_filepath
    #         )

    # ==================================================================================================================
    # Run the point cloud processing step, which adds random error as specified to the noise-free point clouds

    # e.run_steps([Scenario.process_point_cloud])

    # Evaluate the point cloud properties to decide on how to proceed with the reconstruction optimization

    # e.run_steps(Scenario.setup_evaluation)
    # e.run_steps(Scenario.run_evaluation, evaluator_selection="point_density")

    # ==================================================================================================================
    # Perform the full reconstruction optimization for selected scenarios. In theory, only the seed scenario, starting
    # from which the sequential optimization is performed for the other scenarios.

    # geoflow_default_params_for_optim = {
    #     k: glb.geoflow_parameters_default[k]
    #     for k in [k2.split("_", 1)[1] for k2 in glb.geoflow_optim_parameter_space_narrow_2.keys()]
    # }

    # A: Do it the traditional way

    # scenario_to_optimize = "scenario_055"
    # scenario_to_optimize_idx = 55
    # e.run_steps([Scenario.setup_reconstruction_optimization, Scenario.prepare_reconstruction_optimization],
    #             scenarios=scenario_to_optimize)
    # e[scenario_to_optimize_idx].recon_optim.optimizer.probe(
    #     params=geoflow_default_params_for_optim,
    #     lazy=True
    # )
    # e.run_steps(Scenario.run_reconstruction_optimization, scenarios=scenario_to_optimize, init_points=40, n_iter=160)

    # B: Use the newer function

    # e.optimize_reconstruction_params(
    #     sequence=[55],
    #     init_points=40,
    #     n_iter=160,
    #     recon_timeout=600,
    #     probe_parameter_sets=[geoflow_default_params_for_optim],
    #     continue_from_previous_run=True
    # )

    # ------------------------------------------------------------------------------------------------------------------
    # Additionally compute 3D IOU as evaluation metric for the reconstruction optimization for the seed scenario

    # eo = Experiment.load(e[scenario_to_optimize_idx].recon_optim_output_dirpath, load_scenarios=True)
    # eo.run_steps(Scenario.setup_evaluation, lods=["2.2"])
    # eo.run_steps(Scenario.run_evaluation, evaluator_selection=["iou_3d", "complexity", "geoflow_output"])

    # ==================================================================================================================
    # Perform the sequential optimization following the sequence / cascade

    # Calling the following is unnecessary since it's imported from the setup script
    # load_scenario_optimizer_states, next_scenarios, processing_sequence = get_processing_sequence(
    #     n_scenarios=n_scenarios,
    #     seed_scenario=55
    # )

    # A: Only theoretically works if nothing ever crashes

    # processing_sequence = processing_sequence[1:]  # Remove seed scenario (55) from sequence
    # e.optimize_reconstruction_params(
    #     sequence=processing_sequence,
    #     init_points=5,
    #     n_iter=45,
    #     probe_parameter_sets=[geoflow_default_params_for_optim]
    # )

    # B: Same as A, but this is to continue if it already did crash

    # processing_sequence_part = processing_sequence[80:81]
    # e.optimize_reconstruction_params(
    #     sequence=processing_sequence_part,
    #     init_points=5,
    #     n_iter=45,
    #     probe_parameter_sets=[geoflow_default_params_for_optim],
    #     continue_from_previous_run=True
    # )

    # C: This handles crashes by outsourcing the method call to another script

    processing_sequence_part = processing_sequence[99:121]  # start with 1 to skip seed scenario (55)
    continue_previous = False
    n_crashes = 0
    broken_optim_scenarios = {}

    for s_id in processing_sequence_part:
        command = [
            "python",
            Path(r"C:\Users\Florian\OneDrive - TUM\Universität\24\Master's Thesis\Code\city-to-scan-to-city\exp_recon_optim_crasher.py"),
            "--scenario_id",
            str(s_id)
        ]

        scenario_finished = False

        while not scenario_finished:
            if continue_previous and command[-1] != "--continue_previous":
                command.append("--continue_previous")

            print("\nRunning script that calls the reconstruction optimization function.")
            print("Command:")
            print(command)

            try:
                for line in execute_subprocess(command):
                    print(line, end="")

            except Exception as exc:
                n_crashes += 1

                print(f"\nThis is crash number {n_crashes}. The following exception occurred:")
                print(str(exc))

                # Get the name of the optimization scenario that crashed
                print("\nTrying to identify the optimization scenario that caused the exception ...")
                optim_scenario_settings_dirpath = get_most_recently_created_folder(e[s_id].recon_optim_output_dirpath / "02_settings")
                optim_scenario_name = optim_scenario_settings_dirpath.name
                print(f"- Identified as `{optim_scenario_name}`.")

                # Rename any associated folders
                date_time = datetime.today().strftime("%y%m%d-%H%M%S")
                damage_str = "broken" if isinstance(exc, ReconstructionError) else "crashed"
                optim_scenario_name_new = f"{optim_scenario_name}_{damage_str}_{date_time}"

                if isinstance(exc, ReconstructionError):
                    broken_optim_scenarios[e[s_id].name] = optim_scenario_name_new

                print("\nAttemting to rename all associated folders before continuing the optimization ...")
                optim_scenario_settings_dirpath.rename(optim_scenario_settings_dirpath.parent / optim_scenario_name_new)

                for dirname in ["07_reconstruction", "08_evaluation"]:
                    p = e[s_id].recon_optim_output_dirpath / dirname / optim_scenario_name
                    if p.is_dir():
                        p.rename(p.parent / optim_scenario_name_new)

                print(f"- Renamed to {optim_scenario_name_new}")

                continue_previous = True  # set continue_previous such that the command line argument is appended
                # Unless the very first recon optim scenario crashed, in which case we start from scratch.
                if optim_scenario_name == "optim_0000":
                    continue_previous = False

            else:
                scenario_finished = True
                continue_previous = False

    print(f"\nNumber of crashes that were caught: {n_crashes}")
    print(f"\nInstances of outlier vertices in reconstructed models:\n{json.dumps(broken_optim_scenarios, indent=2)}")

    # ==================================================================================================================
    # Selection of the best parameter set

    # scenarios_failed = {}
    # crash_or_catch = "catch"
    # for scenario_id in list(range(0, 110)):  # list(range(26, 110)):
    #     if crash_or_catch == "catch":
    #         try:
    #             e[scenario_id].select_optimal_reconstruction_optimization_scenario()
    #             e[scenario_id].clear_reconstruction_optimization()
    #         except Exception as exc:
    #             print(f"\nScenario {scenario_id} raised an exception:\n")
    #             print(exc)
    #             scenarios_failed[scenario_id] = exc
    #     elif crash_or_catch == "crash":
    #         e[scenario_id].select_optimal_reconstruction_optimization_scenario()
    #         e[scenario_id].clear_reconstruction_optimization()
    #     else:
    #         print("You gotta catch yourself while crashing - or crash.")
    #         break
    #
    # print("\nFinished selecting best parameter sets for all scenarios.")
    # print("The following scenarios raised an exception:\n")
    # print([scid for scid in scenarios_failed.keys()])
    # print("\nThese are the corresponding exceptions:\n")
    # for scid, exc in scenarios_failed.items():
    #     print(f"\nScenario {scid}:\n")
    #     print(exc)

    # Reconstruction

    # e.run_steps(
    #     [
    #         Scenario.load_optimized_reconstruction_params,
    #         Scenario.setup_reconstruction,
    #         Scenario.prepare_reconstruction,
    #         Scenario.run_reconstruction,
    #         Scenario.clear_reconstruction
    #     ],
    #     scenarios=list(range(0, 110))
    # )
