# Genetic Algorithm for Solving the Traveling Salesman Problem

This repository contains a Python implementation of a Genetic Algorithm to solve the Traveling Salesman Problem (TSP). The algorithm aims to find the shortest possible route that visits each city once and returns to the origin city. The implementation includes several steps and functions to read the input data, generate an initial population, select parents, perform crossover and mutation, and evaluate the fitness of the routes.

## File Structure

- `GA - TSP - files-20231103/berlin11_modified.tsp`: The input file containing the coordinates of the cities.

## Classes and Functions

### 1. `Project` Class

The `Project` class encapsulates all the functionalities required to solve the TSP using a Genetic Algorithm.

#### `__init__(self)`
Initializes the `Project` class with the following attributes:
- `fileName`: Name of the TSP data file.
- `result`: List to store the original representation of cities.
- `distance`: List to store distances between cities and shuffled cities.
- `getRes`: List to store all results.
- `parents`: List to store parents for crossover.

#### `makeResult(self)`
Reads the input file and stores the coordinates of the cities in the `result` list.

#### `indexes(self, arr, position)`
Converts city coordinates to their respective indices.

#### `info(self)`
Displays shuffled list of city indices and calculates the distance between two random cities.

#### `fitness(self, arr)`
Calculates the fitness (total distance) of a given route.

#### `fitness2(self, arr)`
Another method to calculate fitness, used for comparing different routes.

#### `population(self, loop)`
Generates an initial population of random routes.

#### `tournament(self)`
Selects two parent routes from the population using tournament selection.

#### `ordered_crossover(self, parent1, parent2)`
Performs ordered crossover between two parent routes to produce offspring.

#### `mutation(self, child1)`
Applies mutation to a given route by reversing a subsequence of cities.

#### `newEpoch(self)`
Generates a new offspring using crossover and mutation.

#### `generatingPop(self, loop)`
Generates a new population of routes.

#### `getRepCity(self, arr)`
Converts indices of cities back to their coordinate representation.

#### `findBetter(self)`
Finds the best route in the current population and generates new populations iteratively.

## Usage

The following is an example of how to use the `Project` class:

```python
# Initialize the project
show = Project()

# Read the input data
show.makeResult()

# Generate initial population
show.population(100)

# Perform tournament selection and generate new populations
show.generatingPop(100)

# Find the best route in the current population
best_route = show.findBetter()
print(best_route)

# Create plots to visualize the results
import matplotlib.pyplot as plt
import numpy as np

def quadratic_function(x):
    return x**2

def inverse_quadratic_function(y):
    return y**2 * 2

def inverse_quadratic_function2(j):
    return j**2 * 3

list1 = show.city  # Best route
list2, list3 = [], []

counter = 0
while counter != len(show.pltCity):
    if len(list2) == 0 and show.pltCity[counter] != list1:
        list2 = show.pltCity[counter]
    if len(list3) == 0 and show.pltCity[counter] != list1 and show.pltCity[counter] != list2:
        list3 = show.pltCity[counter]
    counter += 1

# Get sums for plot
def originalRep(arr):
    res = []
    for j in arr:
        res.append(show.result[j - 1])
    return res

firstCitySum = show.fitness(originalRep(list1))
secondCitySum = show.fitness(originalRep(list2))
thirdCitySum = show.fitness(originalRep(list3))

if secondCitySum > thirdCitySum:
    thirdCitySum = secondCitySum
    secondCitySum = thirdCitySum
    list2, list3 = list3, list2

print(int(firstCitySum[0][0]))

# Generate x values for the original quadratic function
x_values = np.linspace(0, 5, 100)

# Generate y values for the original quadratic function
y_values_quadratic = quadratic_function(x_values)

# Generate y values for the inverse quadratic function
y_values_inverse = inverse_quadratic_function(x_values)
j_values_inverse = inverse_quadratic_function2(x_values)

# Plot the results
plt.plot(x_values, y_values_quadratic, label=f'Sum: {int(firstCitySum[0][0])}; City: {list1}')
plt.plot(x_values, y_values_inverse, label=f'Sum: {int(secondCitySum[0][0])}; City: {list2}')
plt.plot(x_values, j_values_inverse, label=f'Sum: {int(thirdCitySum[0][0])}; City: {list3}')

# Add labels and a legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
