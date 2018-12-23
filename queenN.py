# Team member
# 16ucs218 - Sankalp Wakodikar
# 16ucs182 - Shubham Sharma
# 16ucs113 - Mukul Agrawal


import random
import numpy as np
global N


def queenN(pop_size, percent, maxGen, mut):
    generation = 0
    population = generate_population(pop_size)
    fitness = []
    fitness = fitness_score(population)
    while(not test(fitness, percent) and generation < maxGen):
        generation += 1
        population = newPopulation(
            population, fitness, mut)
        fitness = fitness_score(population)


def newPopulation(population, fitness_score, mut):
    population_Size = len(population)
    new_population = []
    new_population += [selectElite(population, fitness_score)]
    while(len(new_population) < population_Size):
        (mate1, mate2) = select(population, fitness_score)
        new_population += [mutate(crossover(mate1, mate2), mut)]
        if len(new_population) < population_Size:
            new_population += [mutate(crossover(mate2, mate1), mut)]

    return new_population


def mutate(sequence, mut):

    point = random.randint(1, mutate)
    if point == 1:
        rn_1 = random.randint(0, N)
        rn_2 = random.randint(0, N)
        temp = sequence[rn_1]
        sequence[rn_1] = sequence[rn_2]
        sequence[rn_2] = temp
    return sequence


def crossover(mate1, mate2):
    break_point = random.randint(0, len(mate1)-1)
    mutated_arr = []
    for i in range(break_point):
        mutated_arr.append(mate1[i])

    for j in range(break_point, len(mate2)):
        if mate2[j] not in mutated_arr:
            mutated_arr.append(mate2[j])

    if len(mutated_arr) < len(mate1):
        for i in mate2[0:break_point]:
            if i not in mate1[0:break_point]:
                mutated_arr.append(i)


def selectElite(population, fitness_score):
    elite = 0
    for i in range(len(fitness_score)):
        if fitness_score[i] < fitness_score[elite]:
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


def generate_population_element_chrom():
    population_dist_element = np.arange(N)
    np.random.shuffle(population_dist_element)
    # print((population_dist_element))
    return population_dist_element


def generate_population(pop_size):
    randomPopulation = []
    for i in range(pop_size):
        list_seq = generate_population_element_chrom()
        # print((list_seq.tolist()))
        randomPopulation += [list_seq.tolist()]

    return(randomPopulation)


def fitness_score(pop):
    fit_scores = []
    for i in (pop):
        fit_scores.append(isSafe(i))
    # print(fit_scores)
    return fit_scores


def isSafe(dist_sequence):

    sum = []
    diff = []
    # checking for upper diagonal
    for i in range(len(dist_sequence)):
        sum.append(i + dist_sequence[i])
        diff.append(i - dist_sequence[i])
    sum = sorted(sum)
    diff = sorted(diff)
    count_clash = 0

    for i in (range(len(dist_sequence)-1)):
        temp = 0
        temp = sum[i]
        if sum[i+1] == temp:
            count_clash += 1
        temp = diff[i]
        if diff[i+1] == temp:
            count_clash += 1

    if(count_clash == 0):
        print("optimum solution ->", dist_sequence)
        exit(0)

    return (count_clash)


N = 4
population_size = 10000
maxGen = 100000
mut = 100
(queenN(population_size, 0.3, maxGen, mut))
# generate_population(10)
# if no output is generated then the optimal solution has not been reached
# in the given number of generations
