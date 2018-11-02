from random import randint
from simplegeneticalgorithm import SimpleGeneticAlgorithm


min = 0
max = 100000
target = 100

'''
crea un individuo casualmente e lo restituisce
'''
def generate():
    return [randint(0, 10000) for x in range(6)]


'''
prende in input un individuo e ritorna una sua versione mutata casualmente
'''
def mutate(individual_to_mutate):
    pos_to_mutate = randint(0, len(individual_to_mutate) - 1)
    individual_to_mutate[pos_to_mutate] = randint(min, max)
    return individual_to_mutate


'''
prende in input un individuo e ritorna un numero. Maggiore è il numero e migliore è l'individuo
'''
def fitness(individual_to_fit):
    sumfit = 0
    for element in individual_to_fit:
        sumfit += element
    return 1/((abs(target - sumfit))+1)


'''
prende in intput due individui e ritorna un individuo risultato della combinazione genetica dei due
'''
def breed(male, female):
    half = int(len(male) / 2)
    child = male[:half] + female[half:]
    return child


sga = SimpleGeneticAlgorithm(generate,
                             mutate,
                             fitness,
                             breed,
                             population_size=1000,
                             survival_rate=0.5,
                             random_selection_rate=0.1,
                             mutation_rate=0.1)
for x in range(2000):
    sga.evolve(1)
    print(sga.get_population_grade())
    print(str(sga.get_best_individual()) + "->" + str(sum(sga.get_best_individual())))
