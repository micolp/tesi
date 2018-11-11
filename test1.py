from skimage import io

from genetic_engine import GeneticEngine
from genetic_functions_definitions import generate, mutate, fitness, crossover, training_set
import ge_config as cfg
from ge_export import export

ge = GeneticEngine(generate, mutate, fitness, crossover,
                   population_size=cfg.population_size,
                   survival_rate=cfg.survival_rate,
                   random_selection_rate=cfg.random_selection_rate,
                   mutation_rate=cfg.mutation_rate,
                   verbose_level=cfg.verbose_level)

print("Starting...")
ge.evolve(cfg.generation_count)
print("Done.")
print("Applying best pipeline on test image...")
test_image = io.imread('images/test_image.png', as_gray=True)
best = ge.get_elite()[0]
filtered_test = best.process(test_image)
print(filtered_test.max())
print("Exporting results...")
export(ge.get_elite(), filtered_test)
print("Done.")
