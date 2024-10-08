import argparse
import json
from experiment.scenario import Scenario, Experiment
from experiment_setup.exp_utrecht_10_492_594_setup import *


def run_surveys(scenarios: str | list[str]):
    print("\nSetting up surveys ...")
    e.run_steps(Scenario.setup_survey, scenarios=scenarios)

    print("\nPreparing surveys ...")
    e.run_steps(Scenario.prepare_survey, scenarios=scenarios)

    print("\nRunning surveys ...")
    e.run_steps([Scenario.run_survey, Scenario.clear_survey], scenarios=scenarios)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script takes a single ID for the list with unique survey names")
    parser.add_argument("survey_id", type=int, help="Unique survey name list ID")
    args = parser.parse_args()

    e = Experiment(experiment_name, experiment_dirpath, default_config, scenario_settings, scene_parts)

    print("\nSetting up experiment ...")
    e.setup()

    print("\nUnique surveys:")
    print(scenarios_unique_surveys)

    print("\nScenario settings:\n")

    for i, ss in enumerate(scenario_settings):
        print(str(i) + ": " + str(ss))

    print("\nTarget densities and point spacing:\n")
    print(json.dumps(dict(zip(target_densities, point_spacings)), indent=4))

    print(f"\nArgument received as unique survey ID: {args.survey_id}\n")

    run_surveys(scenarios_unique_surveys[int(args.survey_id)])
