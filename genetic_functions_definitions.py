from random import randint, choice

import numpy as np

import p_filters
import ge_config as cfg
import ge_toolkit as tk

training_set = tk.training_set_list[cfg.training_set]()


# crea un individuo casualmente e lo restituisce
def generate():
    pipeline_length = randint(cfg.pipeline_min_filters, cfg.pipeline_max_filters)
    pipeline = p_filters.Pipeline()
    for i in range(pipeline_length - 1):
        pipeline.add_filter(tk.get_random_filter())
    # ogni pipeline ha una soglia come filtro finale
    pipeline.add_filter(choice(cfg.threshold_set)())
    return pipeline


# prende in input un individuo e ritorna una sua versione mutata casualmente
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


# prende in input un individuo (in questo caso una pipeline) e ritorna un numero.
# Maggiore è il numero e migliore è l'individuo
def fitness(individual_to_fit):
    success_count = 0
    fails_count = 0

    # training set è una lista di esempi (ogni esempio è (image, oracle))
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


# prende in input due individui (male, female : due pipeline)
# e ritorna un individuo risultato della combinazione genetica dei due
def crossover(male, female):
    male_length = male.get_length()
    female_length = female.get_length()
    child = male.get_subpipeline(0, male_length // 2) + female.get_subpipeline(female_length // 2)
    return child
