# 2.
import random

# 3. Constants
TARGET_PHRASE = "I love The Tech Academy!" # The target phrase to be matched
POPULATION_SIZE = 250 # Number of individuals in the population
MUTATION_RATE = 0.02 # Probavility of mutation

# 4. Generate initial population
def generate_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.!') for _ in range(len(TARGET_PHRASE)))
        population.append(individual)
    return population

# 5. Calculate fitness score
def calculate_fitness(individual):
    score = 0
    for i in range(len(TARGET_PHRASE)):
        if individual[i] == TARGET_PHRASE[i]:
            score += 1
    return score

# 6. Select parents based on fitness
def select_parents(population):
    parents = []
    for _ in range(2):
        parents.append(max(population, key=calculate_fitness))
    return parents

# 7. Create offspring through crossover 
def crossover(parents):
    offspring = ""
    crossover_point = random.randint(0, len(TARGET_PHRASE) - 1)
    for i in range(len(TARGET_PHRASE)):
        if i <= crossover_point:
            offspring += parents[0][i]
        else:
            offspring += parents[1][i]
    return offspring

# 8. Mutate offspring
def mutate(offspring):
    mutated_offspring = ""
    for i in range(len(offspring)):
        if random.random() < MUTATION_RATE:
            mutated_offspring += random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.!')
        else:
            mutated_offspring += offspring[i]
    return mutated_offspring

# 9a. Main genetic algorithm
def genetic_algorithm():
    population = generate_population()
    generation = 1

    while True:
        print(f"Generation {generation} - Best Fit: {max(population, key=calculate_fitness)}")

        if TARGET_PHRASE in population:
            break

        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parents = select_parents(population)
            offspring = crossover(parents)
            mutated_offspring = mutate(offspring)
            new_population.extend([offspring, mutated_offspring])

        population = new_population
        generation += 1

# 9b. Run the genetic algorithm
genetic_algorithm()