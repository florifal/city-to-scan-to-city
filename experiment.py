import json
import shutil
from datetime import datetime

from experiment.scenario import Scenario, Experiment
from experiment_setup.exp_utrecht_10_492_594_v2_setup import *
from experiment.utils import execute_subprocess, get_most_recently_created_folder
from experiment.reconstruction import ReconstructionError

if __name__ == "__main__":
    load_existing_experiment = True  # set false to apply changes made to the config

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

    s_ids = range(4, 12)
    
    for s_id in s_ids:
        command = [
            "python",
            Path(r"C:\Users\Florian\OneDrive - TUM\Universität\24\Master's Thesis\Code\city-to-scan-to-city\exp_survey_crasher.py"),
            str(s_id)
        ]
    
        print(f"\n[{t}] Running experiment.py for ID {s_id} ...")
        print(f"Command:\n{command}")
        # t = datetime.today().strftime("%y%m%d-%H%M%S")
        # log_filename = f"survey_{s_id}_at_{t}.log"
    
        try:
            for line in execute_subprocess(command):
                pass
            # The log file has around 50 MB, so I decided against keeping it
            # with open(log_filename, "w", encoding="utf-8") as f:
            #     for line in execute_subprocess(command):
            #         f.write(line)
        except Exception as e:
            print("\nError occurred as expected:")
            print(str(e))
        else:
            print("\nNo error occurred unexpectedly.")
    
        print("Continuing with next ID.\n")

    # ==================================================================================================================
    # Merge point cloud swaths independently after running the crashing survey with exp_survey_wrapper.py

    # A) Run complete merging procedure, filtering included

    e.run_steps([
        Scenario.setup_survey,
        Scenario.prepare_survey,
        Scenario.merge_clouds,
        Scenario.clear_survey
    ],
        scenarios=scenarios_unique_surveys
    )

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

    e.run_steps([
        Scenario.setup_survey,
        Scenario.prepare_survey,
        Scenario.clear_survey
    ])
    
    n_density_levels = len(target_densities)
    n_noise_levels = len(error_steps)
    
    for i in [j * n_noise_levels for j in range(n_density_levels)]:
        print(i)
        merged_cloud_textfile_filepath = e.scenarios[f"scenario_{i:03}"].textfile_merged_cloud_path_filepath
        for k in range(1, n_noise_levels):
            print(f" {i + k}")
            shutil.copy2(
                merged_cloud_textfile_filepath,
                e.scenarios[f"scenario_{(i + k):03}"].textfile_merged_cloud_path_filepath
            )

    # ==================================================================================================================
    # Run the point cloud processing step, which adds random error as specified to the noise-free point clouds

    e.run_steps([Scenario.process_point_cloud])

    Evaluate the point cloud properties to decide on how to proceed with the reconstruction optimization

    e.run_steps(Scenario.setup_evaluation)
    e.run_steps(Scenario.run_evaluation, evaluator_selection="point_density")

    # ==================================================================================================================
    # Perform the full reconstruction optimization for selected scenarios. In theory, only the seed scenario, starting
    # from which the sequential optimization is performed for the other scenarios.

    geoflow_default_params_for_optim = {
        k: glb.geoflow_parameters_default[k]
        for k in [k2.split("_", 1)[1] for k2 in glb.geoflow_optim_parameter_space_narrow_2.keys()]
    }

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

    e.optimize_reconstruction_params(
        sequence=[55],
        init_points=40,
        n_iter=160,
        recon_timeout=600,
        probe_parameter_sets=[geoflow_default_params_for_optim],
        continue_from_previous_run=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Additionally compute 3D IOU as evaluation metric for the reconstruction optimization for the seed scenario

    eo = Experiment.load(e[scenario_to_optimize_idx].recon_optim_output_dirpath, load_scenarios=True)
    eo.run_steps(Scenario.setup_evaluation, lods=["2.2"])
    eo.run_steps(Scenario.run_evaluation, evaluator_selection=["iou_3d", "complexity", "geoflow_output"])

    # ==================================================================================================================
    # Perform the sequential optimization following the sequence / cascade

    # Calling the following is unnecessary since it's imported from the setup script
    load_scenario_optimizer_states, next_scenarios, processing_sequence = get_processing_sequence(
        n_scenarios=n_scenarios,
        seed_scenario=55
    )

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

    processing_sequence_part = processing_sequence[113:121]  # start with 1 to skip seed scenario (55)
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
    
            recon_error = False
    
            try:
                for line in execute_subprocess(command):
                    if "ReconstructionError" in line:
                        recon_error = True
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
                damage_str = "broken" if recon_error else "crashed"
                optim_scenario_name_new = f"{optim_scenario_name}_{damage_str}_{date_time}"
    
                if recon_error:
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
    # Clean up after reconstruction optimization

    # Load optimization experiments for all scenarios
    scenarios = list(range(110))
    
    eos = []
    for si in scenarios:
        print(f"\n{si}", end="")
        eos.append(Experiment.load(e[si].recon_optim_output_dirpath, load_scenarios=True))
    
    print("\nRemoving unsuccessfull optimization scenarios ...")
    
    def remove_unsuccessful_optim_scenarios(damage_str_snippet: str = "crash", dest_dirname: str = "graveyard"):
        dest_dirpath = e.recon_optim_dirpath / dest_dirname
        dest_dirpath.mkdir(exist_ok=True)
        for eo in eos:
            print(f"\nOptim experiment for {eo.name}")
            dirpaths = [eo.settings_dirpath, eo.reconstruction_dirpath, eo.evaluation_dirpath]
            for p in dirpaths:
                for f in p.iterdir():
                    if f.is_dir() and damage_str_snippet in f.name:
                        name_new = f"{eo.name}_{p.name}_{f.name}"
                        print(f"- Moving `{f.name}` to `{name_new}`")
                        f = f.rename(f.parent / name_new)
                        shutil.move(f, dest_dirpath)
    
    remove_unsuccessful_optim_scenarios("crash", "graveyard")
    remove_unsuccessful_optim_scenarios("broken", "hospital")
    
    # Reload optimization experiments without the scenarios that were removed
    eos = []
    for si in scenarios:
        print(f"\n{si}", end="")
        eos.append(Experiment.load(e[si].recon_optim_output_dirpath, load_scenarios=True))
    
    for eo in eos:
        eo.run_steps(Scenario.setup_evaluation, lods=["2.2"])
    
    print("\nNumber of optimization scenarios in which Geoflow timed out for each optim experiment:")
    for eo in eos:
        n_opt_sc = len(eo)
        n_timeout = len([s for s in eo.scenarios.values() if s.flag_no_recon_output])
        # Number of optimization scenarios in which zero buildings were reconstructed (must subtract those in which
        # Geoflow timed out for each optim experiment)
        n_zero_buildings = len([s for s in eo.scenarios.values() if s.flag_zero_buildings_reconstructed]) - n_timeout
        print(f"{eo.name}: Timeout {n_timeout} / {n_opt_sc}, zero buildings {n_zero_buildings} / {n_opt_sc}")
    
    # Run additional evaluators to check their results
    for eo in eos:
        eo.run_steps(Scenario.run_evaluation, evaluator_selection=["complexity", "complexity_diff", "geoflow_output"])
    
    # Compute summary statistics for the evaluators whose results are available
    for eo in eos:
        eo.compute_summary_statistics(
            evaluator_selection=["hausdorff", "complexity", "complexity_diff", "geoflow_output"],
            ignore_missing=True
        )

    # ==================================================================================================================
    # Selection of the best parameter set

    scenarios_failed = {}
    crash_or_catch = "catch"
    best_params_filename = "best_parameter_set_10%_min_faces.json"
    scenario_ids = list(range(110, 121))
    
    for scenario_id in scenario_ids:
        if crash_or_catch == "catch":
            try:
                e[scenario_id].select_optimal_reconstruction_optimization_scenario(
                    params_filename=best_params_filename,
                    update_config=True,
                    save_config=True
                )
                e[scenario_id].clear_reconstruction_optimization()
            except Exception as exc:
                print(f"\nScenario {scenario_id} raised an exception:\n")
                print(exc)
                scenarios_failed[scenario_id] = exc
        elif crash_or_catch == "crash":
            e[scenario_id].select_optimal_reconstruction_optimization_scenario()
            e[scenario_id].clear_reconstruction_optimization()
        else:
            print("You gotta catch yourself - or crash.")
            break
    
    print("\nFinished selecting best parameter sets for all scenarios.")
    print("The following scenarios raised an exception:\n")
    print([scid for scid in scenarios_failed.keys()])
    print("\nThese are the corresponding exceptions:\n")
    for scid, exc in scenarios_failed.items():
        print(f"\nScenario {scid}:\n")
        print(exc)

    # ==================================================================================================================
    # Reconstruction

    best_params_filename = "best_parameter_set_10%_min_faces.json"
    scenario_ids = list(range(0, 114)) + list(range(116, 121))
    
    e.run_steps(Scenario.load_optimized_reconstruction_params,
                params_filename=best_params_filename,
                scenarios=scenario_ids)

    e.run_steps(
        [
            Scenario.setup_reconstruction,
            Scenario.prepare_reconstruction,
            Scenario.run_reconstruction,
            Scenario.clear_reconstruction
        ],
        scenarios=scenario_ids
    )

    # For scenarios 114 and 115, Geoflow never finished reconstruction even after hours of running. I decided against
    # investing the time to implement an iterative reconstruction of subsets which are then merged. Instead, I manually
    # identified the next best (Pareto-optimal) optimization scenarios for both. In the following, the reconstruction
    # parameters of scenarios 114 and 115 are updated with the parameters of these next best optimization scenarios.

    next_best_optim_scenarios = {
        114: 26,
        115: 11
    }
    
    for scenario_id, next_best_optim_id in next_best_optim_scenarios.items():
        print(f"\nUpdating reconstruction parameters of scenario {scenario_id} with those "
              f"of optimization scenario {next_best_optim_id} ...")
        e[scenario_id].setup_reconstruction_optimization()
        e[scenario_id].recon_optim.load_optim_experiment()
        e[scenario_id].set_reconstruction_params(
            e[scenario_id].recon_optim.optim_experiment[next_best_optim_id].geoflow_parameters,
            save_config=True
        )
    
        print(f"\nFor scenario {scenario_id}, the following reconstruction parameters from "
              f"{e[scenario_id].recon_optim.optim_experiment[next_best_optim_id].name} were set:")
        print(json.dumps(e[scenario_id].geoflow_parameters, indent=2, ensure_ascii=False))

    e.run_steps(
        [
            Scenario.setup_reconstruction,
            Scenario.prepare_reconstruction,
            Scenario.run_reconstruction,
            Scenario.clear_reconstruction
        ],
        scenarios=list(next_best_optim_scenarios.keys())
    )

    # Scenarios 121 to 142 reuse the optimized reconstruction parameters from scenarios 110 to 120. It is done by
    # reading the reconstruction parameters stored in their configuration, so make sure they are actually stored and
    # saved there: 1. Set load_existing_experiment = True, 2. run the lines where the best parameters are loaded for
    # scenarios 0 to 113 and 116 to 120, 3. run the lines where scenarios 114 and 115 load reconstruction parameters
    # from an alternative (next best) optimization scenario. If step 2 and 3 were performed previously, step 1 should
    # be enough.
    # Because Geoflow always got stuck at some particular buildings during processing and therefore never finished the
    # reconstruction, the footprints of the three (five) buildings most frequently causing this were deleted for these
    # high-density scenarios. The filepaths are therefore updated in the reconstruction config.

    scenario_gets_parameters_from_scenario = zip(
        list(range(121, 143)),
        2 * list(range(110, 121))
    )

    fp_fp_1 = str(input_dirpath / "Footprints" / "footprints_10-492-594_subset_high-density.gpkg")
    fp_fp_2 = str(input_dirpath / "Footprints" / "footprints_10-492-594_subset_high-density_2.gpkg")
    fp_fp_3 = str(input_dirpath / "Footprints" / "footprints_10-492-594_subset_high-density_3.gpkg")
    scenario_uses_footprints_filepath = dict(zip(
        list(range(121, 143)),
        [fp_fp_2] + 10 * [fp_fp_1] + [fp_fp_3] + [fp_fp_2] + 9 * [fp_fp_3]
    ))

    for scenario_id, from_scenario_id in scenario_gets_parameters_from_scenario:
        print(f"\nScenario {scenario_id}: Updating reconstruction footprints filepath "
              f"to `{scenario_uses_footprints_filepath[scenario_id]}` ...")
        e[scenario_id].reconstruction_config["building_footprints_filepath"] = scenario_uses_footprints_filepath[scenario_id]
        print(f"\nScenario {scenario_id}: Applying reconstruction parameters from scenario {from_scenario_id} ...")
        e[scenario_id].set_reconstruction_params(e[from_scenario_id].geoflow_parameters, save_config=True)

    scenario_ids = list(range(135, 143))  # list(range(121, 143))

    e.run_steps(
        [
            Scenario.setup_reconstruction,
            Scenario.prepare_reconstruction,
            Scenario.run_reconstruction,
            Scenario.clear_reconstruction
        ],
        scenarios=scenario_ids
    )

    # ==================================================================================================================
    # Evaluation

    scenario_ids =  list(range(0, 132))
    reevaluate = True
    evaluators = ["geoflow_output", "hausdorff", "complexity", "complexity_diff", "height_diff", "point_density"]

    e.run_steps(Scenario.setup_evaluation, scenarios=scenario_ids)
    for scenario_id in scenario_ids:
        e.run_steps(Scenario.run_evaluation, scenarios=scenario_id, evaluator_selection=evaluators, reevaluate=reevaluate)
        del e[scenario_id].evaluators  # free up memory

    # ==================================================================================================================
    # Re-Evaluation of ComplexityDifferenceEvaluator including input and output numbers of faces

    scenario_ids = list(range(0, 132))  # list(range(0, 121))
    reevaluate = True
    evaluators = ["complexity_diff"]

    e.run_steps(Scenario.setup_evaluation, scenarios=scenario_ids)
    for scenario_id in scenario_ids:
        e.run_steps(Scenario.run_evaluation, scenarios=scenario_id, evaluator_selection=evaluators, reevaluate=reevaluate)
        del e[scenario_id].evaluators  # free up memory

    # ==================================================================================================================
    # Evaluation Summary

    scenario_ids = list(range(0, 132))
    evaluators = ["geoflow_output", "hausdorff", "complexity", "complexity_diff", "height_diff", "point_density"]
    
    e.run_steps(Scenario.setup_evaluation, scenarios=scenario_ids)
    e.run_steps(Scenario.concat_evaluation_results, scenarios=scenario_ids, evaluator_selection=evaluators)
    e.compute_summary_statistics(scenarios=scenario_ids, evaluator_selection=evaluators)
    
    # Get final results of all BuildingsEvaluators concatenated into one table
    evaluators = ["hausdorff", "complexity", "complexity_diff", "height_diff"]
    e.compute_final_results(evaluator_selection=evaluators, scenarios=scenario_ids)
