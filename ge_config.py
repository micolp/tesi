import p_filters as pf

# ----------------------------------------------------------------------------------------------------------------------
# Pipelines/Filters variables
# ----------------------------------------------------------------------------------------------------------------------
pipeline_min_filters = 1
pipeline_max_filters = 2

# ----------------------------------------------------------------------------------------------------------------------
# Genetic algorithm engine variables
# ----------------------------------------------------------------------------------------------------------------------
population_size = 20
survival_rate = 0.3
random_selection_rate = 0.1
mutation_rate = 0.2
exporting_gap = 5
training_set = 'scratches'
verbose_level = 3

# ----------------------------------------------------------------------------------------------------------------------
# Fitness variables
# ----------------------------------------------------------------------------------------------------------------------
success_weight = 1
fails_weight = 2

# ----------------------------------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------------------------------
image_path = 'images/Kitties.jpg'
test_image_path = 'images/Kitties_test.jpg'

# ----------------------------------------------------------------------------------------------------------------------
# Sets
# ----------------------------------------------------------------------------------------------------------------------
edge_detector_set = (pf.Sobel, pf.Roberts, pf.Prewitt, pf.Scharr, pf.Canny)
threshold_set = (pf.ThresholdGlobal, pf.ThresholdGlobal) # pf.ThresholdLocal
morphology_set = (pf.Erode, pf.Dilate, pf.Open, pf.Close, pf.Skeleton) # pf.Thin
misc_set = (pf.Laplacian, pf.Gaussian, pf.Invert) # , pf.Hessian, pf.Frangi

category_set = (edge_detector_set, threshold_set, morphology_set, misc_set)






