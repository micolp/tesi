from random import shuffle

from matplotlib import pyplot as plt

from genetic_engine import GeneticEngine
from genetic_functions_definitions import generate, mutate, fitness, crossover, training_set
import ge_config as cfg
from export import export

ge = GeneticEngine(generate, mutate, fitness, crossover,
                   population_size=cfg.population_size,
                   survival_rate=cfg.survival_rate,
                   random_selection_rate=cfg.random_selection_rate,
                   mutation_rate=cfg.mutation_rate,
                   verbose_level=cfg.verbose_level)

print("Starting...")
ge.evolve(cfg.generation_count)
print("Done...")

export(cfg, ge.get_elite())

print('Retriving best individual of last generation...')
best_pipeline = ge.get_best_individual()
print('Best individual is:')
print(best_pipeline.get_description())
shuffle(training_set)
for example in training_set:
    fig, axs = plt.subplots(2, 2)
    image = example['image']
    oracle = example['oracle']
    filtered = best_pipeline.process(image)
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 1].imshow(filtered, cmap='gray')
    axs[1, 0].text(0.5, 0.5, str(oracle))
    plt.show()





