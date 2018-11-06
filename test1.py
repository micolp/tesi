import numpy as np
from random import randint, choice
from genetic_engine import GeneticEngine
from skimage import io
from matplotlib import pyplot as plt
import p_filters

image = io.imread('images/Kitties.jpg', as_gray=True)
# oracle = io.imread('images/Kitties.png', as_gray=True)/255.0
# oracle = oracle > 0.1

# stiamo creando la nostra griglia valorizzata (matrice di False)
oracle_bool = np.zeros(shape=(6, 10)).astype(bool)
oracle_bool[2][3] = True
oracle_bool[2][4] = True
oracle_bool[2][6] = True
oracle_bool[2][7] = True

# oracle_bool = load_grid()
# fino a qui

# crea matrice di zeri (un'immagine tutta nera della stessa dim dell'immagine originale)
oracle = np.zeros(shape=image.shape)
square_height = (image.shape[0])/oracle_bool.shape[0]
square_width = (image.shape[1])/oracle_bool.shape[1]

# adatta griglia all'immagine di partenza, colorando di bianco (=1) dove il valore è true
for i in range(oracle_bool.shape[0]):
    for j in range(oracle_bool.shape[1]):
        if oracle_bool[i, j]:
            oracle[int(i*square_height):int(i*square_height)+int(square_height),
            int(j*square_width):int(j*square_width)+int(square_width)] = 1

# il minimo/massimo numero di filtri di cui è composta una pipeline
min_filters = 2
max_filters = 10


# crea un individuo casualmente e lo restituisce
def generate():
    pipeline_length = randint(min_filters, max_filters)
    pipeline = p_filters.Pipeline()
    for i in range(pipeline_length-1):
        pipeline.add_filter(get_random_filter())
    # ogni pipeline ha una soglia come filtro finale
    pipeline.add_filter(choice(p_filters.threshold_set)())
    return pipeline


# prende in input un individuo e ritorna una sua versione mutata casualmente
def mutate(individual_to_mutate):
    random_mutation_type = choice(["add", "remove", "replace"])
    random_index = randint(0, individual_to_mutate.get_length()-2)
    if random_mutation_type == "add":
        individual_to_mutate.add_filter(get_random_filter(), random_index)
    elif random_mutation_type == "remove":
        individual_to_mutate.remove_filter(random_index)
    elif random_mutation_type == "replace":
        individual_to_mutate.remove_filter(random_index)
        individual_to_mutate.add_filter(get_random_filter(), random_index)
    return individual_to_mutate


# prende in input un individuo (in questo caso una pipeline) e ritorna un numero.
# Maggiore è il numero e migliore è l'individuo
def fitness(individual_to_fit):
    filtered_image = individual_to_fit.process(image)
    filtered_image = filtered_image.astype(bool)
    # conta pixel dell'itersezione (bianchi nei quadrati bianchi)
    success_sum = np.sum(np.logical_and(filtered_image, oracle))
    # conta i pixel bianchi fuori dei quadrati bianchi
    fails_sum = np.sum(np.logical_and(np.logical_not(oracle), filtered_image))
    return success_sum/(fails_sum+1)


# prende in intput due individui (male, female : due pipeline)
# e ritorna un individuo risultato della combinazione genetica dei due
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
                   population_size=10,
                   survival_rate=0.3,
                   random_selection_rate=0.1,
                   mutation_rate=0.15)

print("Starting...")

for i in range(5):
    print("Current generation: " + str(i))
    ge.evolve(1)

print("Done...")

best_pipeline = ge.get_best_individual()

print("Processing...")

filtered_image = best_pipeline.process(image, normalize=True, verbose=True)

print("Processed!")

print(best_pipeline.get_description())

fig, axs = plt.subplots(3, 2)

oracle.shape = image.shape

test_image = io.imread('images/Kitties_test.jpg', as_gray=True)
test_filtered = best_pipeline.process(test_image)

axs[0, 0].imshow(image, cmap='gray')
axs[0, 1].imshow(oracle, cmap='gray')
axs[1, 0].imshow(filtered_image, cmap='gray')
axs[1, 1].imshow(np.logical_and(filtered_image, oracle), cmap='gray')
axs[2, 0].imshow(test_image, cmap='gray')
axs[2, 1].imshow(test_filtered, cmap='gray')

plt.show()
