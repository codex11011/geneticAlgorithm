

import numpy as np
import random


def Knapsack(values, weights, max_value, pop_size, mut, maxGen, percent):
    generation = 0
    population_generated = generate_population(values, pop_size)
    fitness_score = popfitness_score(
        population_generated, values, weights, max_value)
    # print(fitness_score)
    while(not test(fitness_score, percent) and generation < maxGen):
        generation += 1
        population_generated = newPopulation(
            population_generated, fitness_score, mut)
        fitness_score = popfitness_score(
            population_generated, values, weights, max_value)

    return selectElite(population_generated, fitness_score)


def generate_population(values, pop_size):
    length = len(values)
    population = [[random.randint(0, 1) for i in range(length)]
                  for j in range(pop_size)]
    return population


def popfitness_score(pop, values, weights, max_value):
    fitness = []
    for i in range(len(pop)):
        weight = 0
        volume = max_value+1
        while (volume > max_value):
            weight = 0
            volume = 0
            ones = []
            for j in range(len(pop[i])):
                if pop[i][j] == 1:
                    volume += values[j]
                    weight += weights[j]
                    ones += [j]
                if volume > max_value:
                    pop[i][ones[random.randint(0, len(ones)-1)]] = 0
        fitness += [weight]
    return fitness


def test(fit_score, rate):
    maxCount = max_count_fitness_score(fit_score)
    if float(maxCount)/float(len(fit_score)) >= rate:
        return True
    else:
        return False


def max_count_fitness_score(fitness_score):
    values = set(fitness_score)
    maxCount = 0
    for i in values:
        if maxCount < fitness_score.count(i):
            maxCount = fitness_score.count(i)
    return maxCount


def newPopulation(population, fitness_score, mut):
    population_Size = len(population)
    new_population = []
    new_population += [selectElite(population, fitness_score)]
    while(len(new_population) < population_Size):
        (mate1, mate2) = select(population, fitness_score)
        new_population += [mutate(crossover(mate1, mate2), mut)]

    return new_population


def selectElite(population, fitness_score):
    elite = 0
    for i in range(len(fitness_score)):
        if fitness_score[i] > fitness_score[elite]:
            elite = i
    return population[elite]


def select(population, fitness_score):
    # Roulette Wheel Selection algorithm
    size = len(population)
    sum_fitness_score = sum(fitness_score)
    point = random.randint(0, sum_fitness_score)
    tempSum = 0
    mate1 = []
    fit1 = 0
    for i in range(size):
        tempSum += fitness_score[i]
        if tempSum >= point:
            mate1 = population.pop(i)
            fit1 = fitness_score.pop(i)
            break
    tempSum = 0
    point = random.randint(0, sum(fitness_score))
    for i in range(len(population)):
        tempSum += fitness_score[i]
        if tempSum >= point:
            mate2 = population[i]
            population += [mate1]
            fitness_score += [fit1]
            return (mate1, mate2)


def crossover(mate_1, mate_2):
    break_point = random.randint(0, len(mate_1)-1)
    return mate_1[:break_point]+mate_2[break_point:]


def mutate(subject_gene, mutate):
    for i in range(len(subject_gene)):
        point = random.randint(1, mutate)
        if point == 1:
            subject_gene[i] = bool(subject_gene[i]) ^ 1
    return subject_gene


volume = [1, 3, 1, 3, 2, 4, 3]
weights = [30, 20, 50, 70, 50, 10, 60]
max_volume = 6
pop_size = 10
print(Knapsack(volume, weights, max_volume, pop_size, 100, 100, 0.5))
