from skimage import io

from genetic_engine import GeneticEngine
from genetic_functions_definitions import generate, mutate, fitness, crossover
import ge_config as cfg
from ge_export import export
import ge_toolkit as tk

ge = GeneticEngine(generate, mutate, fitness, crossover,
                   population_size=cfg.population_size,
                   survival_rate=cfg.survival_rate,
                   random_selection_rate=cfg.random_selection_rate,
                   mutation_rate=cfg.mutation_rate,
                   )

print("Starting...")
test_image = io.imread('images/test_image.png', as_gray=True)
while True:
    ge.evolve(cfg.exporting_gap)
    print("Applying best pipeline on test image...")
    best = ge.get_elite()[0]
    filtered_test = best.process(test_image)
    filtered_test = filtered_test.astype(bool)
    filtered_test_tiled, positives_negatives_image, positives, negatives = tk.filtered_tile_by_tile(best, test_image)
    print("Exporting results...")
    try:
        export(ge.get_elite(),
               filtered_test,
               ge.generation_count,
               filtered_test_tiled,
               positives_negatives_image,
               positives,
               negatives)
    except Exception as error:
        print("An error occurred while exporting results:")
        print(str(error))
    print("Export done.")
