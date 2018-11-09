from random import randint, choice

import numpy as np
from matplotlib import pyplot as plt

from genetic_engine import GeneticEngine
import p_filters
import ge_config as cfg
import ge_toolkit as tk

training_set = tk.training_set_list[cfg.training_set()]


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
    fitness_values = []
    success_sums = []
    oracle_success_sums = []
    fails_sums = []
    oracle_fails_sums = []
    # training set è una lista di esempi (ogni esempio è (image, oracle))
    for example in training_set:
        image = example['image']
        oracle = example['oracle']
        filtered_image = individual_to_fit.process(image)
        filtered_image = filtered_image.astype(bool)
        # conta pixel dell'itersezione (bianchi nei quadrati bianchi)
        success_sum = np.sum(np.logical_and(filtered_image, oracle)) # intersezione matrici
        success_sums.append(success_sum)
        # conta i pixel bianchi fuori dei quadrati bianchi
        fails_sum = np.sum(np.logical_and(np.logical_not(oracle), filtered_image))
        fails_sums.append(fails_sum)
    global_success_sum = sum(success_sums)
    global_fails_sum = sum(fails_sums)
    return (global_success_sum + 1) / (global_fails_sum + 1)

# prende in input due individui (male, female : due pipeline)
# e ritorna un individuo risultato della combinazione genetica dei due
def crossover(male, female):
    male_length = male.get_length()
    female_length = female.get_length()
    child = male.get_subpipeline(0, male_length // 2) + female.get_subpipeline(female_length // 2)
    return child


ge = GeneticEngine(generate, mutate, fitness, crossover,
                   population_size=cfg.population_size,
                   survival_rate=cfg.survival_rate,
                   random_selection_rate=cfg.random_selection_rate,
                   mutation_rate=cfg.mutation_rate)

print("Starting...")

for i in range(cfg.generation_count):
    print("Current generation: " + str(i))
    ge.evolve(1)
    print(ge.get_population_grade())
    print(ge.get_best_individual_fitness())

print("Done...")


def show_recap():
    print('Retriving best individual of last generation...')
    best_pipeline = ge.get_best_individual()
    print('Best individual is:')
    print(best_pipeline.get_description())
    for example in training_set:
        fig, axs = plt.subplots(2, 2)
        image = example['image']
        filtered = best_pipeline.process(image)
        axs[0, 0].imshow(image, cmap='gray')
        axs[0, 1].imshow(example['oracle'], cmap='gray')
        axs[1, 0].imshow(filtered, cmap='gray')
        axs[1, 1].imshow(np.logical_and(filtered, example['oracle']), cmap='gray')
        plt.show()


show_recap()


