from random import randint, choice
from genetic_engine import GeneticEngine
from skimage import io
from matplotlib import pyplot
import p_filters

# il minimo/massimo numero di filtri di cui è composta una pipeline
min_filters = 2
max_filters = 5


# crea un individuo casualmente e lo restituisce
def generate():
    pipeline_length = randint(min_filters, max_filters)
    pipeline = p_filters.Pipeline()
    for i in range(pipeline_length):
        random_category = choice(p_filters.category_set)
        random_filter_class = choice(random_category)
        random_filter = random_filter_class()
        pipeline.add_filter(random_filter)
    return pipeline


# prende in input un individuo e ritorna una sua versione mutata casualmente
def mutate(individual_to_mutate):
    return individual_to_mutate


# prende in input un individuo e ritorna un numero. Maggiore è il numero e migliore è l'individuo
def fitness(individual_to_fit):
    return individual_to_fit.get_length()


# prende in intput due individui (male, female : due pipeline) e ritorna un individuo risultato della combinazione genetica dei due
def crossover(male, female):
    male_length = male.get_length()
    female_length = female.get_length()
    child = male.get_subpipeline(0, male_length // 2) + female.get_subpipeline(female_length // 2)
    return child


ge = GeneticEngine(generate,
                   mutate,
                   fitness,
                   crossover,
                   population_size=1000,
                   survival_rate=0.5,
                   random_selection_rate=0.1,
                   mutation_rate=0.1)

print("Starting...")

for x in range(2000):
    ge.evolve(1)

print("Done...")

best_pipeline = ge.get_best_individual()

image = io.imread('images/Picture.png', as_gray=True)
image = best_pipeline.process(image, verbose=True)

print(best_pipeline.get_description())

io.imshow(image)
pyplot.show()


