# 2.
import numpy as np

# 3. Define the distance matrix
distance_matrix = np.array([
    [9, 0, 1, 6, 4],
    [1, 0, 3, 5, 8],
    [2, 4, 0, 9, 3],
    [3, 7, 1, 0, 2],
    [5, 4, 3, 2, 0]
])

# 4. Number of cities
num_cities = distance_matrix.shape[0]

# 5. Number of ants
num_ants = 26

# 6.
def ant_colony_optimization(num_iterations):
    # Initialize pheromone level matrix
    pheromone_level = np.ones((num_cities, num_cities))

    # 7. Initialize heuristic information matrix
    heuristic_info = 1 / (distance_matrix + np.finfo(float).eps) # Avoid division by zero

    # 8. Alpha and beta parameters
    alpha = 1.0 # Pheromone importance
    beta = 2.0 # Heuristic importance

    # 9. Initialize best path and distance
    best_distance = float('inf')
    best_path = []
    best_iteration = -1 #Initializing the best_iteration variable

    for iteration in range(num_iterations):
        # Initialize ant paths and distances
        ant_paths = np.zeros((num_ants, num_cities), dtype=int)
        ant_distances = np.zeros(num_ants)

        # 10.
        for ant in range(num_ants):
            # Choose the starting city randomly
            current_city = np.random.randint(num_cities)
            visited = [current_city]

            # Construct the path
            for _ in range(num_cities - 1):
                # Calculate the selection probabilities
                selection_probs = (pheromone_level[current_city] ** alpha) * (heuristic_info[current_city] ** beta)
                selection_probs[np.array(visited)] = 0 # Set selection probabilities of visited cities to 0

                # Choose the next city based on the selection probabilities
                next_city = np.random.choice(np.arange(num_cities), p=(selection_probs / np.sum(selection_probs)))

                # Update the path and visited list
                ant_paths[ant, _+1] = next_city
                visited.append(next_city)

                # Update the distance
                ant_distances[ant] += distance_matrix[current_city, next_city]

                # Update the current city
                current_city = next_city

            # Update the distance to return to the starting city
            ant_distances[ant] += distance_matrix[current_city, ant_paths[ant, 0]]
        
        # 11. Update the pheromone level based on the ant paths
        pheromone_level *= 0.5 # Evaporation
        for ant in range(num_ants):
            for city in range(num_cities - 1):
                pheromone_level[ant_paths[ant, city], ant_paths[ant, city+1]] += 1 / ant_distances[ant]
            pheromone_level[ant_paths[ant, -1], ant_paths[ant, 0]] += 1 / ant_distances[ant]

        # 12. Update the best path and distance if a better solution is found
        min_distance_idx = np.argmin(ant_distances)
        if ant_distances[min_distance_idx] < best_distance:
            best_distance = ant_distances[min_distance_idx]
            best_path = ant_paths[min_distance_idx]
            best_iteration = iteration # Update the best_iteration

    return best_path, best_distance, best_iteration

# 13a. Run the Ant Colony Optimization algorithm
num_iterations = 350 # Number of iterations
best_path, best_distance, best_iteration = ant_colony_optimization(num_iterations)

# 13b. Display the best path and distance
print("Here is the best path:", best_path)
print("Here is the best distance:", best_distance)
print("Iteration with the best_distance:", best_iteration) # New line