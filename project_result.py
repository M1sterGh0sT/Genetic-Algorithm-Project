import random
import math


class Project:
    # create variables to store all important data
    def __init__(self):
        # file names
        # berlin52  berlin11_modified  fl417  nrw1379 pr2392
        self.fileName = 'berlin11_modified'

        # store original representation of cities
        self.result = []

        # store distances between cities and shuffled cities
        self.distance = []

        # store all results
        self.getRes = []

        # store parents
        self.parents = []


    def makeResult(self):
        with open(f'GA - TSP - files-20231103/{self.fileName}.tsp', 'r') as file:
            content = file.readlines()

        # make main list with sublist
        for j in content[6:-2]:
            parts = j.split()
            if len(parts) == 3:
                self.result.append([float(parts[1]), float(parts[2])])

    # convert city to index representation
    def indexes(self, arr, position):
        city = []
        for j in arr[position::]:
            if j in self.result:
                city.append(self.result.index(j) + 1)
        return city



    def info(self):
        # check length
        with open(f'GA - TSP - files-20231103/{self.fileName}.tsp', 'r') as file:
            content = file.readlines()

        # list for storing len of cities
        shuffledList = []

        for j in content[6:-2:]:
            shuffledList.append(j.split()[0])

        # shuffle len list
        random.shuffle(shuffledList)
        print(f'Info: {shuffledList}')

        # create 2 variables to get randomly 2 coordinates
        self.randomCoordinates_1 = random.choice(shuffledList)
        shuffledList.pop(shuffledList.index(self.randomCoordinates_1))

        # second value
        self.randomCoordinates_2 = int(random.choice(shuffledList))


        def computeDistance(x, y):
            # result val of 2 cities
            betweenVal = math.sqrt((math.pow(y[0] - x[0], 2) + (math.pow(y[1] - x[1], 2))))

            print(f'\n\nDistance between city: {self.randomCoordinates_1} and city: {self.randomCoordinates_2} -> {betweenVal}\n\n')

        # take 2 random cities and compute distance between them
        return computeDistance(self.result[int(self.randomCoordinates_1) - 1], self.result[self.randomCoordinates_2 - 1])





    def fitness(self, arr):
        self.cityRep = []

        firstElement = arr[0]
        distance = 0
        lastVal = math.sqrt(math.pow(arr[0][0] - arr[-1][0], 2) + math.pow(arr[0][1] - arr[-1][1], 2))
        for j in arr[1::]:
            distance += math.sqrt(math.pow(j[0] - firstElement[0], 2) + math.pow(j[1] - firstElement[1], 2))
            firstElement = j
        resDistance = distance + lastVal

        self.cityRep.append([resDistance] + arr)
        return self.cityRep


    def fitness2(self, arr):
        self.result2 = []


        firstElement = arr[0]
        distance = 0
        lastVal = math.sqrt(math.pow(arr[0][0] - arr[-1][0], 2) + math.pow(arr[0][1] - arr[-1][1], 2))
        for j in arr[1::]:
            distance += math.sqrt(math.pow(j[0] - firstElement[0], 2) + math.pow(j[1] - firstElement[1], 2))
            firstElement = j
        resDistance = distance + lastVal

        self.getRes.append([resDistance] + arr)


        self.result2.append([resDistance] + arr)

        return self.result2



    # main variable "getRes"


    def population(self, loop):
        self.cityPopulation = []

        res2 = self.result.copy()

        for j in range(loop+1):
            while True:
                random.shuffle(res2)
                get = self.fitness(res2)
                if get not in self.cityPopulation:
                    break
            self.cityPopulation.append(self.fitness(res2))

        return self.cityPopulation





    # get 2 parents
    def tournament(self):
        storePlayers = []
        population = self.cityPopulation.copy()

        for j in range(3):
            storePlayers.append(population.pop(population.index(random.choice(population))))

        # print(f'Store {storePlayers}')

        justSumRes = []
        for j in storePlayers:
            justSumRes.append(j[0])

        counter = 0
        findParents = []
        while counter != 2:
            findParents.append(justSumRes.pop(justSumRes.index(min(justSumRes))))
            counter += 1

        createFullParents = []
        for j in findParents:
            for s in storePlayers:
                if j == s[0]:
                    createFullParents.append(s)
        parents = createFullParents


        self.parents = createFullParents
        # print(createFullParents)

        # save sum
        # print(findParents)
        return self.parents



    def ordered_crossover(self, parent1, parent2):


        length = len(parent1)

        # Select two random crossover points
        crossover_point1 = random.randint(0, length - 1)
        crossover_point2 = random.randint(0, length - 1)

        # Make sure crossover_point2 is greater than crossover_point1
        crossover_point1, crossover_point2 = min(crossover_point1, crossover_point2), max(crossover_point1,
                                                                                          crossover_point2)

        # Initialize offspring with the genetic material of parents
        offspring1 = [-1] * length
        offspring2 = [-1] * length

        # Copy the genetic material between the crossover points from parents to offspring
        offspring1[crossover_point1:crossover_point2 + 1] = parent1[crossover_point1:crossover_point2 + 1]
        offspring2[crossover_point1:crossover_point2 + 1] = parent2[crossover_point1:crossover_point2 + 1]

        # Fill in the remaining positions in offspring with genetic material from the other parent
        pointer1, pointer2 = 0, 0
        for i in range(length):
            if offspring1[i] == -1:
                while parent2[pointer2] in offspring1:
                    pointer2 += 1
                offspring1[i] = parent2[pointer2]
                pointer2 += 1

            if offspring2[i] == -1:
                while parent1[pointer1] in offspring2:
                    pointer1 += 1
                offspring2[i] = parent1[pointer1]
                pointer1 += 1



        #print(f'\n\nParents: {parent1}, {parent2}')
        # return f'Child: {offspring1}, {offspring2}'
        return offspring1


    def mutation(self, child1):
        choiceNum = len(child1)

        possibleRange = [j for j in range(choiceNum)]
        num1 = random.choice(possibleRange)
        possibleRange.pop(possibleRange.index(num1))
        num2 = random.choice(possibleRange)

        if num1 > num2:
            num1, num2 = num2, num1

        center = child1[num1:num2]
        left = child1[0:num1]
        right = child1[num2::]

        res = left + center[::-1] + right
        return res


    def newEpoch(self):
        self.tournament()
        child = self.ordered_crossover(self.indexes(show.parents[0][0], 0), self.indexes(show.parents[0][0], 1))
        new = self.mutation(child)
        return new


    def generatingPop(self, loop):
        self.newPopulation = []
        for j in range(loop):
            res = self.newEpoch()
            if res not in self.newPopulation:
                self.newPopulation.append(res)

        return self.newPopulation


    # take indexes of cities and get X and Y representation
    def getRepCity(self, arr):
        city = []

        counter = 0
        while counter != len(arr):
            inner = []
            for j in arr[counter]:
                inner.append(self.result[j - 1])

            city.append(inner)
            counter += 1

        # print(f'Arr: {arr} \nRes:{self.result} \nRep:{city}')
        return city


    def findBetter(self):

        # create new population

        population = self.generatingPop(100)
        indexCity = self.getRepCity(population)

        counter = 0
        allSums = []
        while counter != len(indexCity):
            get = self.fitness2(indexCity[counter])

            allSums.append(get)
            counter += 1

        index = 0
        smallest = allSums[0][0][0]

        # find smallest from range and take original rep and position

        for j in allSums:
            for s in j:
                if smallest >= s[0]:
                    smallest = s[0]
                    compare = s
                    index = allSums.index(j)


        self.city = []
        for j in allSums[index][0][1:]:
            if j in self.result:

                self.city.append(self.result.index(j) + 1)


        self.pltCity = []

        counter = 0
        while counter != len(allSums):
            inner = []
            for j in allSums[counter][0][1:]:
                if j in self.result:
                    inner.append(self.result.index(j) + 1)
            if inner not in self.pltCity:
                self.pltCity.append(inner)
            counter += 1



        print(f'Smallest: {self.pltCity}')

        return f'{allSums[index]} \n{allSums[index][0][1:]} \n{self.city}'





show = Project()
show.makeResult()
# show.info()




print('\n\n')
show.population(100)
print(show.tournament())
show.generatingPop(100)
print(show.findBetter())
# print(len(show.generatingPop(100)))





# create plot
import matplotlib.pyplot as plt
import numpy as np



def quadratic_function(x):
    return x**2

# Define the inverse function
def inverse_quadratic_function(y):
    return y**2 * 2

def inverse_quadratic_function2(j):
    return j**2 * 3




list1 = show.city   # smallest
list2 = []
list3 = []



counter = 0
while counter != len(show.pltCity):
    if len(list2) == 0:
        if show.pltCity[counter] != list1:
            list2 = show.pltCity[counter]

    if len(list3) == 0:
        if show.pltCity[counter] != list1 and show.pltCity[counter] != list2:
                    list3 = show.pltCity[counter]
    counter += 1



# Get sums
def originalRep(arr):
    res = []
    for j in arr:
        res.append(show.result[j - 1])
    return res

# get sums for plot
firstCitySum = show.fitness(originalRep(list1))
secondCitySum = show.fitness(originalRep(list2))
thirdCitySum = show.fitness(originalRep(list3))

if secondCitySum > thirdCitySum:
    thirdCitySum = secondCitySum
    secondCitySum = thirdCitySum
    list2 = list3
    list3 = list2

print(int(firstCitySum[0][0]))


# Generate x values for the original quadratic function
x_values = np.linspace(0, 5, 100)

# Generate y values for the original quadratic function
y_values_quadratic = quadratic_function(x_values)

# Generate y values for the inverse quadratic function
y_values_inverse = inverse_quadratic_function(x_values)

j_values_inverse = inverse_quadratic_function2(x_values)


# Plot the original quadratic function
plt.plot(x_values, y_values_quadratic, label=f'Sum: {int(firstCitySum[0][0])}; City: {list1}')

# Plot the inverse quadratic function
plt.plot(x_values, y_values_inverse,  label=f'Sum: {int(secondCitySum[0][0])}; City: {list2}')

plt.plot(x_values, j_values_inverse,  label=f'Sum: {int(thirdCitySum[0][0])}; City: {list3}')

# Add labels and a legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()




# made by _m.gh0st