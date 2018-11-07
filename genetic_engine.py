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
                 mutation_rate=0.01):  #prob di mutazione casuale per ogni individuo di elite
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

    '''
    per ogni generazione:
        seleziono una percentuale di individui da far sopravvivere tramite funzione di fitness
        aggiungo casualmente elementi da scartare per aggiungere varietà genetica
        applico delle mutazioni casuali agli individui sopravvissuti
        faccio riprodurre gli individui sopravvissuti
    '''
    def evolve(self, generation_count):
        for g in range(generation_count):
            self.sort_population_by_fitness()

            survived_population_size = int(len(self.population) * self.survival_rate)
            parents = self.population[0:survived_population_size]

            # randomly add other individuals to promote genetic diversity
            for individual in self.population[survived_population_size:]:
                if self.random_selection_rate > random():
                    parents.append(individual)

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

            # mutate some children
            for i in range(len(children)):
                if self.mutation_rate > random():
                    children[i] = self.mutate(children[i])
                    children[i].fitness_value = None

            self.population = parents + children

    def sort_population_by_fitness(self):
        for individual in self.population:
            if not individual.fitness_value:
                individual.fitness_value = self.fitness(individual)
        self.population.sort(key=lambda individual: individual.fitness_value, reverse=True)

    def get_best_individual(self):
        self.sort_population_by_fitness()
        return self.population[0]

    def get_best_individual_fitness(self):
        return self.get_best_individual().fitness_value

    def get_population_grade(self):
        return sum([self.fitness(individual) for individual in self.population])/len(self.population)

