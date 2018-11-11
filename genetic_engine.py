from random import random
from random import randint


class GeneticEngine:
    def __init__(self,
                 generate,  #crea individui pop iniziale
                 mutate,  #crea mutazione genetica casuale
                 fitness,  #valore più alto = individuo migliore
                 breed,  #crea individui gen successive a partire da madre e padre
                 population_size=100,  #numero individui
                 survival_rate=0.2,  #numero individui migliori da usare come genitori per gen successiva
                 random_selection_rate=0.05,  #prob di sopravvivenza di individuo non di elite
                 mutation_rate=0.01,  #prob di mutazione casuale per ogni individuo di elite
                 verbose_level=0):
        self.generate = generate
        self.mutate = mutate
        self.fitness = fitness
        self.breed = breed
        self.population_size = population_size
        self.survival_rate = survival_rate
        self.random_selection_rate = random_selection_rate
        self.mutation_rate = mutation_rate
        # Crea una popolazione di individui casualmente
        self.population = [self.generate() for i in range(self.population_size)]
        # Assegna un valore di fitness nullo ad ogni individuo
        for individual in self.population:
            individual.fitness_value = None
        self.verbose_level = verbose_level

    '''
    per ogni generazione:
        calcolo la fitness della mia popolazione
        seleziono una percentuale di individui da far sopravvivere in base alla fitness
        aggiungo casualmente elementi presi da quelli da scartare per aggiungere varietà genetica
        faccio riprodurre gli individui sopravvissuti
        applico delle mutazioni casuali agli individui appena generati
    '''
    def evolve(self, generation_count):
        verbose = self.verbose_level
        for g in range(generation_count):
            if verbose >= 1: print("Current generation: " + str(g))
            if verbose >= 1: print("----------------------------------------------------------------------------------")
            if verbose >= 2: print("Population size:" + str(self.population_size))
            if verbose >= 2: print("Calculating individuals fitness...")
            self.compute_population_fitness()

            if verbose >= 1: print("Mean population fitness is: " + str(sum([individual.fitness_value for
                                                                        individual in
                                                                        self.population])/len(self.population)))
            if verbose >= 2: print("Selecting elite individuals...\n"
                                   "Survival rate is: " + str(self.survival_rate))
            self.sort_population_by_fitness()
            survived_population_size = int(len(self.population) * self.survival_rate)
            parents = self.population[0:survived_population_size]
            if verbose >= 1: print("Best individual fitness is: " + str(parents[0].fitness_value))

            if verbose >= 2: print("Randomly add other individuals in elite to promote genetic diversity...\n"
                                   "Chanses to be added are: " + str(self.random_selection_rate))
            for individual in self.population[survived_population_size:]:
                if self.random_selection_rate > random():
                    parents.append(individual)

            if verbose >= 2: print("Coupling elite individuals...")
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

            if verbose >= 2: print("Randomly mutate some children to promote genetic diversity...\n"
                                   "Chances to mutate are: " + str(self.mutation_rate))
            for i in range(len(children)):
                if self.mutation_rate > random():
                    children[i] = self.mutate(children[i])
                    children[i].fitness_value = None

            self.population = parents + children
            if verbose >= 1: print("----------------------------------------------------------------------------------")

    def compute_population_fitness(self):
        for individual in self.population:
            if self.verbose_level >= 3: print(". ", end="", flush=True)
            if individual.fitness_value is None:
                individual.fitness_value = self.fitness(individual)
        if self.verbose_level >= 3: print("")

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
