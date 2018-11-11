import p_filters as pf
import json


# ----------------------------------------------------------------------------------------------------------------------
# Pipelines/Filters variables
# ----------------------------------------------------------------------------------------------------------------------
pipeline_min_filters = 1
pipeline_max_filters = 2

# ----------------------------------------------------------------------------------------------------------------------
# Genetic algorithm engine variables
# ----------------------------------------------------------------------------------------------------------------------
population_size = 10
survival_rate = 0.3
random_selection_rate = 0.1
mutation_rate = 0.15
generation_count = 1
training_set = 'daniel'

# ----------------------------------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------------------------------
image_path = 'images/Kitties.jpg'
test_image_path = 'images/Kitties_test.jpg'

# ----------------------------------------------------------------------------------------------------------------------
# Sets
# ----------------------------------------------------------------------------------------------------------------------
edge_detector_set = (pf.Sobel, pf.Roberts, pf.Prewitt, pf.Scharr, pf.Canny)
threshold_set = (pf.ThresholdGlobal, pf.ThresholdGlobal) #, ThresholdLocal)
morphology_set = (pf.Erode, pf.Dilate, pf.Open, pf.Close, pf.Skeleton)#, Thin
misc_set = (pf.Laplacian, pf.Gaussian, pf.Invert)# Frangi, Hessian,

category_set = (edge_detector_set, threshold_set, morphology_set, misc_set)


def export_config(file_path):
    # creo lista con tutti i nomi a partire dalla lista dei filtri
    edge_detector_filters_names = [filter().get_filter_name() for filter in edge_detector_set]
    threshold_set_filters_names = [filter().get_filter_name() for filter in threshold_set]
    morphology_set_filters_names = [filter().get_filter_name() for filter in morphology_set]
    misc_set_filters_names = [filter().get_filter_name() for filter in misc_set]
    category_set_names = []

    for category in category_set:
        for variable, value in globals().items():
            if category == value:
                category_set_names.append(variable)

    genetic_engine = {'pipelines': {'min-size': pipeline_min_filters,
                                    'max-size': pipeline_max_filters},
                      'genetic-algorithm': {'training-set': training_set,
                                            'population-size': population_size,
                                            'survival-rate': survival_rate,
                                            'random-selection-rate': random_selection_rate,
                                            'mutation-rate': mutation_rate,
                                            'generation-count': generation_count},
                      'sets': {'used-categories': category_set_names,
                               'used-edge-detectors': edge_detector_filters_names,
                               'used-thresholds': threshold_set_filters_names,
                               'used-morphology': morphology_set_filters_names,
                               'used-misc': misc_set_filters_names}}

    with open(file_path, 'w') as config_file: #'export/ge.config'
        json.dump(genetic_engine, config_file)
        

# export_config()


