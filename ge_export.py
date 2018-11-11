from ge_config import export_config
import json
import datetime
import os


def export(result):
    now = datetime.datetime.now()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    subdir_name = now.strftime("export_%Y-%m-%d_%H-%M-%S/")
    export_dir = "exports/" + subdir_name
    config_filename = "genetic_engine_parameters.json"
    filepath = os.path.join(current_dir, export_dir, config_filename)
    os.mkdir(os.path.join(current_dir, export_dir))
    config = export_config()
    with open(filepath, 'w') as config_file:
        json.dump(config, config_file, indent=4)

    result_list = []
    for individual in result:
        pipeline = individual.filters_list
        pipeline_dict = {
            "fitness": str(individual.fitness_value),
            "pipeline": []
        }
        for p_filter in pipeline:
            filterdict = {
                "class": str(p_filter.__class__),
                "class-variables": str(p_filter.__dict__)
            }
            pipeline_dict["pipeline"].append(filterdict)
        result_list.append(pipeline_dict)
    result_filename = "elite.json"
    filepath = os.path.join(current_dir, export_dir, result_filename)
    with open(filepath, 'w') as result_file:
        json.dump(result_list, result_file, indent=4)
