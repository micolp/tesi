import json
import datetime
import os

from skimage import io

import ge_config as cfg


def export_config():
    # create a list with all the names starting from the filter list
    edge_detector_filters_names = [filter().get_filter_name() for filter in cfg.edge_detector_set]
    threshold_set_filters_names = [filter().get_filter_name() for filter in cfg.threshold_set]
    morphology_set_filters_names = [filter().get_filter_name() for filter in cfg.morphology_set]
    misc_set_filters_names = [filter().get_filter_name() for filter in cfg.misc_set]
    category_set_names = []

    for category in cfg.category_set:
        for variable, value in globals().items():
            if category == value:
                category_set_names.append(variable)

    genetic_engine = {'pipelines': {'min-size': cfg.pipeline_min_filters,
                                    'max-size': cfg.pipeline_max_filters},
                      'genetic-algorithm': {'training-set': cfg.training_set,
                                            'population-size': cfg.population_size,
                                            'survival-rate': cfg.survival_rate,
                                            'random-selection-rate': cfg.random_selection_rate,
                                            'mutation-rate': cfg.mutation_rate},
                      'fitness-values': {
                          'success-weight': cfg.success_weight,
                          'fails-weight': cfg.fails_weight,
                      },
                      'sets': {'used-categories': category_set_names,
                               'used-edge-detectors': edge_detector_filters_names,
                               'used-thresholds': threshold_set_filters_names,
                               'used-morphology': morphology_set_filters_names,
                               'used-misc': misc_set_filters_names}}

    return genetic_engine


def export(result, test_image, generation_count, filtered_tiled):
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
            "generation-count": generation_count,
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

    test_image_filename = "test.png"
    filepath = os.path.join(current_dir, export_dir, test_image_filename)
    io.imsave(filepath, test_image*255)

    test_tiled_image_filename = "test_tile.png"
    filepath = os.path.join(current_dir, export_dir, test_tiled_image_filename)
    io.imsave(filepath, filtered_tiled)

