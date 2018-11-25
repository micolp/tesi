import json
import datetime
import os

from skimage import io

import ge_config as cfg

oracle_positives = []
oracle_positives_filename = "images/test_oracle_positives.coords"
with open(oracle_positives_filename, 'r') as oracle_positives_file:
    oracle_positives = oracle_positives_file.readlines()

oracle_negatives = []
oracle_negatives_filename = "images/test_oracle_negatives.coords"
with open(oracle_negatives_filename, 'r') as oracle_negatives_file:
    oracle_negatives = oracle_negatives_file.readlines()



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


def export(result, test_image, generation_count, filtered_tiled, positives_negatives_image, positives, negatives):
    now = datetime.datetime.now()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    subdir_name = now.strftime("export_%Y-%m-%d_%H-%M-%S" + str(generation_count) + "/")
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

    positives_negatives_filename = "positives_negatives.png"
    filepath = os.path.join(current_dir, export_dir, positives_negatives_filename)
    io.imsave(filepath, positives_negatives_image)

    result_filename = "positives.coords"
    filepath = os.path.join(current_dir, export_dir, result_filename)
    with open(filepath, 'w') as positives_file:
        for positive in positives:
            positives_file.write(str(positive['x'])+','+str(positive['y']) + '\n')

    with open(filepath, 'r') as positives_file:
        test_positives = positives_file.readlines()
    real_positives = []
    false_positives = []
    for test_positive in test_positives:
        if test_positive in oracle_positives:
            real_positives.append(test_positive)
        else:
            false_positives.append(test_positive)

    result_filename = "negatives.coords"
    filepath = os.path.join(current_dir, export_dir, result_filename)
    with open(filepath, 'w') as negatives_file:
        for negative in negatives:
            negatives_file.write(str(negative['x']) + ',' + str(negative['y']) + '\n')

    with open(filepath, 'r') as negatives_file:
        test_negatives = negatives_file.readlines()
    real_negatives = []
    false_negatives = []
    for test_negative in test_negatives:
        if test_negative in oracle_negatives:
            real_negatives.append(test_negative)
        else:
            false_negatives.append(test_negative)

    result_filename = "real_positives.coords"
    filepath = os.path.join(current_dir, export_dir, result_filename)
    with open(filepath, 'w') as real_positives_file:
        for real_positive in real_positives:
            real_positives_file.write(real_positive)

    result_filename = "false_positives.coords"
    filepath = os.path.join(current_dir, export_dir, result_filename)
    with open(filepath, 'w') as false_positives_file:
        for false_positive in false_positives:
            false_positives_file.write(false_positive)

    result_filename = "real_negatives.coords"
    filepath = os.path.join(current_dir, export_dir, result_filename)
    with open(filepath, 'w') as real_negatives_file:
        for real_negative in real_negatives:
            real_negatives_file.write(real_negative)

    result_filename = "false_negatives.coords"
    filepath = os.path.join(current_dir, export_dir, result_filename)
    with open(filepath, 'w') as false_negatives_file:
        for false_negative in false_negatives:
            false_negatives_file.write(false_negative)

    result_filename = "reacap.txt"
    filepath = os.path.join(current_dir, export_dir, result_filename)
    with open(filepath, 'w') as recap_file:
        recap_file.write('oracle_positives: ' + str(len(oracle_positives)) + '\n')
        recap_file.write('oracle_negatives: ' + str(len(oracle_negatives)) + '\n')
        recap_file.write('real_positives: ' + str(len(real_positives)) + '\n')
        recap_file.write('false_positives: ' + str(len(false_positives)) + '\n')
        recap_file.write('real_negatives: ' + str(len(real_negatives)) + '\n')
        recap_file.write('false_negatives: ' + str(len(false_negatives)) + '\n')
