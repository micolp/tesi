from random import random
from random import randint
import ge_config as cfg
from ge_toolkit import log


class GeneticEngine:
    def __init__(self,
                 # creates individuals of the initial population
                 generate,
                 # creates random genetic mutation
                 mutate,
                 # higher value = better individual
                 fitness,
                 # creates individuals of the next generations starting from mother and father
                 breed,
                 # number of individuals
                 population_size=100,
                 # number of best individuals to use as parents for the next generation
                 survival_rate=0.2,
                 # probability of survival of a not élite individual
                 random_selection_rate=0.05,
                 # probability of random mutation for each élite individual
                 mutation_rate=0.01,
                 ):
        self.generate = generate
        self.mutate = mutate
        self.fitness = fitness
        self.breed = breed
        self.population_size = population_size
        self.survival_rate = survival_rate
        self.random_selection_rate = random_selection_rate
        self.mutation_rate = mutation_rate
        self.generation_count = 0
        # Creates a random population of individuals
        self.population = [self.generate() for i in range(self.population_size)]
        # Assigns a null fitness value to each individual
        for individual in self.population:
            individual.fitness_value = None

    '''
    for each generation:
        calculate the population fitness
        select a percentage of individuals to survive based on fitness
        add random elements from those unwrapped to add genetic diversity
        breed the surviving individuals
        apply random mutations to newly generated individuals
    '''
    def evolve(self, generations_to_iterate):
        for g in range(generations_to_iterate):
            log("Current generation: " + str(self.generation_count), 1)
            log("----------------------------------------------------------------------------------", 1)
            log("Population size:" + str(self.population_size), 2)
            log("Calculating individual's fitness...", 2)
            self.compute_population_fitness()

            log("Mean population fitness is: " + str(sum([individual.fitness_value for
                                                                        individual in
                                                                        self.population])/len(self.population)), 1)
            log("Selecting elite individuals...\n"
                                   "Survival rate is: " + str(self.survival_rate), 2)
            self.sort_population_by_fitness()
            survived_population_size = int(len(self.population) * self.survival_rate)
            parents = self.population[0:survived_population_size]
            log("Best individual fitness is: " + str(parents[0].fitness_value), 1)

            log("Randomly add other individuals to elite to promote genetic diversity...\n"
                                   "Chances to be added are: " + str(self.random_selection_rate), 2)
            for individual in self.population[survived_population_size:]:
                if self.random_selection_rate > random():
                    parents.append(individual)

            log("Breeding elite individuals...", 2)
            # crossover parents to create children
            parents_length = len(parents)
            desired_length = len(self.population) - parents_length
            children = []
            while len(children) < desired_length:
                male_index = randint(0, parents_length - 1)
                female_index = randint(0, parents_length - 1)
                if male_index != female_index:
                    male = parents[male_index]
                    female = parents[female_index]
                    child = self.breed(male, female)
                    child.fitness_value = None
                    children.append(child)

            log("Randomly mutate some children to promote genetic diversity...\n"
                                   "Chances to mutate are: " + str(self.mutation_rate), 2)
            for i in range(len(children)):
                if self.mutation_rate > random():
                    children[i] = self.mutate(children[i])
                    children[i].fitness_value = None

            self.population = parents + children
            self.generation_count += 1
            log("----------------------------------------------------------------------------------", 1)

    # prints a dot for each computed individual
    def compute_population_fitness(self):
        for individual in self.population:
            if cfg.verbose_level >= 3: print(". ", end="", flush=True)
            if individual.fitness_value is None:
                individual.fitness_value = self.fitness(individual)
        if cfg.verbose_level >= 3: print("")

    def sort_population_by_fitness(self):
        self.population.sort(key=lambda individual: individual.fitness_value, reverse=True)

    def get_best_individual(self):
        return self.population[0]

    def get_best_individual_fitness(self):
        return self.get_best_individual().fitness_value

    def get_population_grade(self):
        return sum([individual.fitness_value for individual in self.population])/len(self.population)

    def get_population(self):
        return self.population

    def get_elite(self):
        survived_population_size = int(len(self.population) * self.survival_rate)
        parents = self.population[0:survived_population_size]
        return parents
