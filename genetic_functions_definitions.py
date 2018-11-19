from random import randint, choice

import numpy as np

import p_filters
import ge_config as cfg
import ge_toolkit as tk

training_set = tk.training_set_list[cfg.training_set]()


# creates an individual randomly and returns it
def generate():
    pipeline_length = randint(cfg.pipeline_min_filters, cfg.pipeline_max_filters)
    pipeline = p_filters.Pipeline()
    for i in range(pipeline_length - 1):
        pipeline.add_filter(tk.get_random_filter())
    # every pipeline has a threshold as a final filter
    pipeline.add_filter(choice(cfg.threshold_set)())
    return pipeline


# takes as input an individual and returns its randomly mutated version
def mutate(individual_to_mutate):
    random_mutation_type = choice(["add", "remove", "replace"])
    random_index = randint(0, individual_to_mutate.get_length() - 2)
    if random_mutation_type == "add":
        individual_to_mutate.add_filter(tk.get_random_filter(), random_index)
    elif random_mutation_type == "remove":
        individual_to_mutate.remove_filter(random_index)
    elif random_mutation_type == "replace":
        individual_to_mutate.remove_filter(random_index)
        individual_to_mutate.add_filter(tk.get_random_filter(), random_index)
    return individual_to_mutate


# takes as input an individual (in this case a filters pipeline) and returns a number.
# the greater the number, the better the individual is
def fitness(individual_to_fit):
    success_count = 0
    fails_count = 0

    # training set is a list of examples (each example is (image, oracle))
    for example in training_set:
        image = example['image']
        oracle = example['oracle']  # è true o false
        filtered_image = individual_to_fit.process(image)
        filtered_image = filtered_image.astype(bool)
        sum = np.sum(filtered_image)
        if oracle == True:
            success_count += sum
        elif oracle == False:
            fails_count += sum
    return cfg.success_weight * success_count - cfg.fails_weight * fails_count


# takes as input two individuals (male, female: two pipelines)
# and returns an individual result of the genetic combination of the two
def crossover(male, female):
    male_length = male.get_length()
    female_length = female.get_length()
    child = male.get_subpipeline(0, male_length // 2) + female.get_subpipeline(female_length // 2)
    return child
