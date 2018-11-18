from random import random
from random import randint
import ge_config as cfg
from logger import log


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
        # Crea una popolazione di individui casualmente
        self.population = [self.generate() for i in range(self.population_size)]
        # Assegna un valore di fitness nullo ad ogni individuo
        for individual in self.population:
            individual.fitness_value = None

    '''
    per ogni generazione:
        calcolo la fitness della mia popolazione
        seleziono una percentuale di individui da far sopravvivere in base alla fitness
        aggiungo casualmente elementi presi da quelli da scartare per aggiungere varietà genetica
        faccio riprodurre gli individui sopravvissuti
        applico delle mutazioni casuali agli individui appena generati
    '''
    def evolve(self, generations_to_iterate):
        verbose = cfg.verbose_level
        for g in range(generations_to_iterate):
            log("Current generation: " + str(self.generation_count), 1)
            log("----------------------------------------------------------------------------------", 1)
            log("Population size:" + str(self.population_size), 2)
            log("Calculating individual's fitnesses...", 2)
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

            log("Coupling elite individuals...", 2)
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
