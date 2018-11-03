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

    '''
    per ogni generazione:
        seleziono una percentuale di individui da far sopravvivere tramite funzione di fitness
        aggiungo casualmente elementi da scartare per aggiungere varietà genetica
        applico delle mutazioni casuali agli individui sopravvissuti
        faccio riprodurre gli individui sopravvissuti
    '''
    def evolve(self, generation_count):
        for g in range(generation_count):
            graded = [(self.fitness(x), x) for x in self.population]
            graded = [x[1] for x in sorted(graded, key=lambda pipeline: pipeline[0], reverse=True)]
            survived_population_size = int(len(graded) * self.survival_rate)
            parents = graded[0:survived_population_size]

            # randomly add other individuals to promote genetic diversity
            for individual in graded[survived_population_size:]:
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
                    children.append(child)

            # mutate some children
            for i in range(len(children)):
                if self.mutation_rate > random():
                    children[i] = self.mutate(children[i])

            self.population = parents + children

    def get_graded_population(self):
        graded = [(self.fitness(x), x) for x in self.population]
        graded = [x[1] for x in sorted(graded, key=lambda pipeline: pipeline[0], reverse=True)]
        return graded

    def get_best_individual(self):
        graded = [(self.fitness(x), x) for x in self.population]
        graded = [x[1] for x in sorted(graded, key=lambda pipeline: pipeline[0], reverse=True)]
        return graded[0]

    def get_population_grade(self):
        return sum([self.fitness(individual) for individual in self.population])/len(self.population)
