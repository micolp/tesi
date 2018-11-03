import numpy as np
from random import randint, choice
from genetic_engine import GeneticEngine
from skimage import io
from matplotlib import pyplot as plt
import p_filters

image = io.imread('images/Kitties.jpg', as_gray=True)
oracle = io.imread('images/Kitties.png', as_gray=True)/255.0
oracle = oracle > 0.1

# il minimo/massimo numero di filtri di cui è composta una pipeline
min_filters = 2
max_filters = 10


# crea un individuo casualmente e lo restituisce
def generate():
    pipeline_length = randint(min_filters, max_filters)
    pipeline = p_filters.Pipeline()
    for i in range(pipeline_length):
        pipeline.add_filter(get_random_filter())
    return pipeline


# prende in input un individuo e ritorna una sua versione mutata casualmente
def mutate(individual_to_mutate):
    random_mutation_type = choice(["add", "remove", "replace"])
    random_index = randint(0, individual_to_mutate.get_length()-1)
    if random_mutation_type == "add":
        individual_to_mutate.add_filter(get_random_filter(), random_index)
    elif random_mutation_type == "remove":
        individual_to_mutate.remove_filter(random_index)
    elif random_mutation_type == "replace":
        individual_to_mutate.remove_filter(random_index)
        individual_to_mutate.add_filter(get_random_filter(), random_index)
    return individual_to_mutate


# prende in input un individuo (in questo caso una pipeline) e ritorna un numero. Maggiore è il numero e migliore è l'individuo
def fitness(individual_to_fit):
    filtered_image = individual_to_fit.process(image)
    filtered_image = filtered_image > 0.5
    success_sum = np.sum(np.logical_and(filtered_image, oracle))
    fails_sum = np.sum(np.logical_xor(oracle, filtered_image))
    return success_sum/(fails_sum+1)


# prende in intput due individui (male, female : due pipeline) e ritorna un individuo risultato della combinazione genetica dei due
def crossover(male, female):
    male_length = male.get_length()
    female_length = female.get_length()
    child = male.get_subpipeline(0, male_length // 2) + female.get_subpipeline(female_length // 2)
    return child


def get_random_filter():
    random_category = choice(p_filters.category_set)
    random_filter_class = choice(random_category)
    return random_filter_class()


ge = GeneticEngine(generate,
                   mutate,
                   fitness,
                   crossover,
                   population_size=20,
                   survival_rate=0.3,
                   random_selection_rate=0.3,
                   mutation_rate=0.3)

print("Starting...")

for i in range(30):
    print("Current generation: " + str(i))
    ge.evolve(1)

print("Done...")

best_pipeline = ge.get_best_individual()

print("Processing...")

filtered_image = best_pipeline.process(image, normalize=True, verbose=True)

print("Processed!")

print(best_pipeline.get_description())

fig, axs = plt.subplots(2, 2)

oracle.shape = image.shape

axs[0, 0].imshow(image, cmap='gray')
axs[0, 1].imshow(oracle, cmap='gray')
filtered_image = filtered_image > 0.5
axs[1, 0].imshow(filtered_image, cmap='gray')
axs[1, 1].imshow(np.logical_and(filtered_image, oracle), cmap='gray')

# print(type(filtered_image))
# print(filtered_image.shape)
# print(oracle.shape)
#
# print(image.max())
# print(image.min())
# print(oracle.max())
# print(oracle.min())
# print(filtered_image.max())
# print(filtered_image.min())
# print((oracle - filtered_image).max())
# print(abs(oracle - filtered_image).min())

plt.show()