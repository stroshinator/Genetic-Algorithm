"""
AI Project 6 - Verbal Arithmetic Solution with GA and Wisdom Of Crowds
Author: Drew Stroshine
"""
#****************************** LIBRARY IMPORTS ***************************************************#
from numpy.random import shuffle
from random import randint
# from tkinter import *
import time
import matplotlib.pyplot as plt

#*************************** FUNCTION DEFINITIONS *************************************************#

####################################################################################################
#   SET UP ALPHAMETIC PROBLEM
#       * This funciton will modify the following sample data
#       * B = 10
#       * OPERAND1: SEND
#       * OPERAND2: MORE
#       * RESULT: MONEY
#   - Find all unique letters in puzzle and save them in a list
####################################################################################################
def setUpProb():
  # GENERATE UNIQUE LETTERS IN PUZZLE
  # Scan through each element of operands
  for i in range(len(operands)):
    for j in range(len(operands[i])):
      buffer = operands[i][j]
      # If letter at this index is not already in uniqLetters, append it to list
      if buffer not in uniqLetters:
        uniqLetters.append(buffer)
        
  # Scan through each element of result
  for i in range(len(result)):
    buffer = result[i]
    # If letter at this index is not already in uniqLetters, append it to list
    if buffer not in uniqLetters:
        uniqLetters.append(buffer)
        
  # If length of uniqLetters is less than B, append empty cells to make equal
  if len(uniqLetters) < B:
    for i in range(B - len(uniqLetters)):
      uniqLetters.append('#' + str(i))
  
  # EXTRACT FIRST LETTERS FROM OPERANDS AND RESULTS
  # Scan through each list in operands
  for i in range(len(operands)):
    # Append first letter of each list to firstLetters list
    firstLetters.append(operands[i][0])
    
  # Append first letter of result to firstLetters list
  firstLetters.append(result[0])


####################################################################################################
#   CREATE POPULATION OF RANDOM SOLUTIONS
#   - Each chromosome in the population will be a unique random combination of all the unique
#     letters in the puzzle; make sure to have empty cells if number of unique letters is less than
#     base and it must follow the alphametic constraints such as a 0 cannot be the value of the
#     first letter in a operand or result
#   - Ex. [S, N, M, #, Y, E, #, R, D, O] where number of unique letters is 8 and base is 10
#   - This function will take population size as an argument and generate the 2D list by reference
####################################################################################################
def createPpl(pplSize):
  # Declare temp
  temp = []
  
  # Generate a population based on pplSize
  for i in range(pplSize):
    # Shuffle elements of uniqLetters
    shuffle(uniqLetters)
    # Set temp equal to copy of shuffled uniqLetters
    temp = list(uniqLetters)
    # Confirm that temp is not already in ppl or breaks alphametic constraints
    # If it doesn't meet constraints, reshuffle temp until constraints are met
    while ((temp in ppl) or (temp[0] in firstLetters)):
      shuffle(temp)
    
    # Append temp to ppl
    ppl.append(list(temp))


####################################################################################################
#   EVALUATE FITNESS OF POPULATION
#   - This function takes each chromosome and decodes the puzzle using the chromosome as a key
#   - The fitness function compares the sum of the decoded operands to the decoded result
#   - FITNESS = ABS(SUM(decodedOPERANDS) - decodedRESULT)
#   - If the fitness is 0, then the chrosome is the solution and return this result immediately
#   - Otherwise, the closer to 0 the fitter the chromosome
#   - The fitness of each chromosome is saved to a list
####################################################################################################
def evalFitness():
  # GENERATE LIST OF FITNESS FOR EACH CHROMOSOME  
  for chromosome in ppl:
    # Set key equal to the current chromosome  
    key = list(chromosome)
    
    # DECODE EACH OPERAND AND DETERMINE NUMBER VALUE
    # Declare decodedOperands list  
    decodedOperands = []
    # Scan through each operand of operands
    for i in range(len(operands)):
      tempVal = 0
      # Scan through each letter of each operand
      for j in range(len(operands[i])):
        # Confirm that letter is not an empty slot
        tempVal += key.index(operands[i][j]) * (B**((len(operands[i])-1) - j))
          # Determine place value of decoded letter 
          
      # Add tempVal to decodedOperands list
      decodedOperands.append(tempVal)
    
    # DECODE RESULTS AND DETERMINE NUMBER VALUE
    tempVal = 0
    for i in range(len(result)):
      tempVal += key.index(result[i]) * (B**((len(result)-1) - i))
        
    # Set decodedResult equal to tempVal
    decodedResult = tempVal  
    
    # CALCULATE AND STORE FITNESS OF EACH CHROMOSOME FROM DECODED OPERANDS AND RESULT
    tempFit = abs(sum(decodedOperands) - decodedResult)
    pplFitness.append(tempFit)


####################################################################################################
#   CALCULATE FITNESS
#   - This function is similar to evalFitness but just gives fitness of individual chromosome.
####################################################################################################
def calcFit(key):
  # DECODE EACH OPERAND AND DETERMINE NUMBER VALUE
  # Declare decodedOperands list  
  decodedOperands = []
  # Scan through each operand of operands
  for i in range(len(operands)):
    tempVal = 0
    # Scan through each letter of each operand
    for j in range(len(operands[i])):
      # Determine place value of decoded letter
      tempVal += key.index(operands[i][j]) * (B**((len(operands[i])-1) - j))
             
    # Add tempVal to decodedOperands list
    decodedOperands.append(tempVal)
  
  # DECODE RESULTS AND DETERMINE NUMBER VALUE
  tempVal = 0
  for i in range(len(result)):
    # Determine place value of decoded letter
    tempVal += key.index(result[i]) * (B**((len(result)-1) - i))
      
  # Set decodedResult equal to tempVal
  decodedResult = tempVal  
  
  # CALCULATE AND STORE FITNESS OF EACH CHROMOSOME FROM DECODED OPERANDS AND RESULT
  tempFit = abs(sum(decodedOperands) - decodedResult)
  return tempFit

####################################################################################################
#   SELECT PARENTS
#   - This function selects parents based on a fitness threshold 
#   - The fitness threshold is determined by computing metrics like the average fitness of the 
#     population from the fitness list
####################################################################################################
def selectParents():
  # Randomly select two chromosomes with higher than average fitness and store as parents
  while len(parents) < 2:
    # Create random index  
    i = randint(0, len(pplFitness)-1)
    # Confirm chromosome at that index has smaller (better) value than average fittness
    while ((pplFitness[i] >= avgFit) or (ppl[i] in parents)):
      i = randint(0, len(pplFitness)-1)
    # Add selected chromosome to parents list
    parents.append(list(ppl[i]))

####################################################################################################
#   CROSSOVER
#   - This function crossovers the contents of two parents to create an offspring
#   - Possibly use the same method as in the TSP crossover method
#   - Make sure all alphametic rules are followed
####################################################################################################
def createOffspring():
  # Declare uniqKey 
  offspring.append(parents[0][0])
  uniqKey = list(uniqLetters)
  uniqKey.remove(parents[0][0])
  
  # Combine elements of parents into one offspring
  daIdx = randint(0,1)
  
  # If daIdx == 0 go in ascending order
  if daIdx == 0:  
    for i in range(1, len(parents[0])):
      # Check whether index is even 
      if (i % 2) == 0:
        # Add to offspring the value of the first parent at that index if it is in uniqKey
        if parents[0][i] in uniqKey:
          offspring.append(parents[0][i])
          uniqKey.remove(parents[0][i])
        # Add to offspring the value of the first parent at that index if it is in uniqKey
        elif parents[1][i] in uniqKey:
          offspring.append(parents[1][i])
          uniqKey.remove(parents[1][i])
        # Otherwise just add to offspring the first element in uniqKey
        else:
          offspring.append(uniqKey.pop(0))
      
      # Otherwise index is odd
      else:
        # Add to offspring the value of the first parent at that index if it is in uniqKey
        if parents[1][i] in uniqKey:
          offspring.append(parents[1][i])
          uniqKey.remove(parents[1][i])
        # Add to offspringthe value of the first parent at that index if it is in uniqKey
        elif parents[0][i] in uniqKey:
          offspring.append(parents[0][i])
          uniqKey.remove(parents[0][i])
        # Otherwise just add to offspring the first element in uniqKey
        else:
          offspring.append(uniqKey.pop(0))
    
  # Otherwise go in descending order
  else:
    for i in range(len(parents[0])-1, 0, -1):
      # Check whether index is even 
      if (i % 2) == 0:
        # Add to offspring the value of the first parent at that index if it is in uniqKey
        if parents[0][i] in uniqKey:
          offspring.append(parents[0][i])
          uniqKey.remove(parents[0][i])
        # Add to offspring the value of the first parent at that index if it is in uniqKey
        elif parents[1][i] in uniqKey:
          offspring.append(parents[1][i])
          uniqKey.remove(parents[1][i])
        # Otherwise just add to offspring the first element in uniqKey
        else:
          offspring.append(uniqKey.pop(0))
      
      # Otherwise index is odd
      else:
        # Add to offspring the value of the first parent at that index if it is in uniqKey
        if parents[1][i] in uniqKey:
          offspring.append(parents[1][i])
          uniqKey.remove(parents[1][i])
        # Add to offspringthe value of the first parent at that index if it is in uniqKey
        elif parents[0][i] in uniqKey:
          offspring.append(parents[0][i])
          uniqKey.remove(parents[0][i])
        # Otherwise just add to offspring the first element in uniqKey
        else:
          offspring.append(uniqKey.pop(0))      
    


####################################################################################################
#   CROSSOVER
#   - This function crossovers the contents of two parents to create an offspring
#   - Possibly use the same method as in the TSP crossover method
#   - Make sure all alphametic rules are followed
#   - Exchange the offspring with 
####################################################################################################
def createRandOffspring():
  # Declare uniqKey
  uniqKey = list(uniqLetters)
  
  # Initialize offspring with all '*'
  for i in range(len(parents[0])):
    offspring.append('*')
  
  # Create offspring with random values from parents
  while len(uniqKey) != 0:
    # Create random index to get parent
    randP = randint(0, 1)
    # Create random index to get value
    randV = randint(0, len(parents[0])-1)
    
    # If offspring[0] is still empty
    # Confirm that all non firstLetters have not already been used
    if offspring[0] == '*':
      # Determine how many non firstLetters are still in uniqKey
      fLttrsLeft = 0
      for i in uniqKey:
        if i not in firstLetters:
          fLttrsLeft += 1
          
      # If there is only one non firstLetter in uniqKey
      # Set first index of offspring equal to it and remove from uniqKey
      if fLttrsLeft == 1:
        for i in uniqKey:
          if i not in firstLetters:
            offspring[0] = i
            uniqKey.remove(i)          
    
    
    # Find value in parents still in uniqKey
    searchCount = 0
    while (parents[randP][randV] not in uniqKey) or (offspring[randV] != '*'):
      # Generate new random indices      
      randP = randint(0, 1)
      randV = randint(0, len(parents[0])-1)
      
      # Add one to search count      
      searchCount += 1
      
      # Stop loop after so many iterations
      if searchCount >= 10000:
        for e in range(1, len(offspring)):
          if offspring[e] == '*':
            offspring[e] = uniqKey.pop()
            break
        break
      
    # Add value to offspring and remove from uniqKey
    if searchCount < 10000:
      offspring[randV] = parents[randP][randV]
      uniqKey.remove(parents[randP][randV])
    
    

####################################################################################################
#   MUTATION
#   - Picks two random numbers and exchanges the cells at the indices of these two random numbers
#     at a random chromosome within the population
#   - Make sure all alphametic rules are followed 
####################################################################################################
def mutate():
  # Generate random indices  
  i = randint(0, len(ppl)-1)
  j = randint(0, len(ppl[0])-1)
  k = randint(0, len(ppl[0])-1)
  
  # If one or both of the random indices is 0
  if j == 0 or k == 0:
     # Only switch  if both of the values are not in firstLetters
    if ppl[i][j] not in firstLetters and ppl[i][k] not in firstLetters:
      tempVal = ppl[i][j]
      ppl[i][j] = ppl[i][k]
      ppl[i][k] = tempVal
  # Otherwise go ahead and switch
  else:
    tempVal = ppl[i][j]
    ppl[i][j] = ppl[i][k]
    ppl[i][k] = tempVal


####################################################################################################
#   WISDOM OF THE CROWDS
#   - This function implements the wisdom of the crowds technique to enhance the affects of the GA
####################################################################################################
def wisdomOfCrowds():
  favorIdx = []
  wiseUniqKey = list(uniqLetters)
  
  # Initialize favorIdx and wiseOffspring
  for i in range(len(wiseUniqKey)):
    favorIdx.append(0)
    wiseOffspring.append(0)
  
  
  # CALCULATE POPULARITY OF EACH VALUE AT EACH INDEX
  allFavors = []
  for i in range(len(ppl[0])):
    # Reset favorIdx
    for m in range(len(favorIdx)):
      favorIdx[m] = 0
    
    # Determine most popular values of ppl at current index i
    for j in pplFitness:
      if j < avgFit/2:
        favorIdx[wiseUniqKey.index(ppl[pplFitness.index(j)][i])] += 1
    
    # Add the favorIdx for this index to allFavors
    allFavors.append(list(favorIdx))
  
  # ASSIGN BEST VALUES TO INDICES IN WISE OFFSPRING  
  # Determine best value at first index 
  usedIdxs = []
  maxIdx = allFavors[0].index(max(allFavors[0]))
  wiseOffspring[0] = wiseUniqKey[maxIdx]
  
  # Track index and value used
  usedIdxs.append(0)
  wiseUniqKey[maxIdx] = '*'
  
  # Repeat for remaining indices
  for i in range(1, len(wiseUniqKey)):
    # Create random index
    x = randint(1, len(allFavors)-1)
    
    # Confirm index has not been used
    while x in usedIdxs:
      x = randint(1, len(allFavors)-1)
    
    maxIdx = allFavors[x].index(max(allFavors[x]))
    
    # Confirm value has not already been used
    while wiseUniqKey[maxIdx] == '*':
      allFavors[x][maxIdx] = -1
      maxIdx = allFavors[x].index(max(allFavors[x]))
    
    # Append best value at max index to wiseOffspring
    wiseOffspring[x] = wiseUniqKey[maxIdx]
    
    # Track index and value used
    usedIdxs.append(x)
    wiseUniqKey[maxIdx] = '*'  
  

####################################################################################################
# DISPLAY SOLUTION
#   - Display decoded solution
####################################################################################################  
def displaySolution(bestAns):
  # DECODE EACH OPERAND AND DETERMINE NUMBER VALUE
  # Declare decodedOperands list  
  decodedOperands = []
  # Scan through each operand of operands
  for i in range(len(operands)):
    tempVal = 0
    # Scan through each letter of each operand
    for j in range(len(operands[i])):
      # Determine place value of decoded letter
      tempVal += bestAns.index(operands[i][j]) * (B**((len(operands[i])-1) - j))
             
    # Add tempVal to decodedOperands list
    decodedOperands.append(tempVal)
  
  # DECODE RESULTS AND DETERMINE NUMBER VALUE
  tempVal = 0
  for i in range(len(result)):
    # Determine place value of decoded letter
    tempVal += bestAns.index(result[i]) * (B**((len(result)-1) - i))
      
  # Set decodedResult equal to tempVal
  decodedResult = tempVal
  
  # DISPLAY SOLUTION
  # Operands portion
  strListOperands = []
  for i in operands:
    for j in i:
      strListOperands.append(j)
    if i == operands[len(operands)-1]:
      strListOperands.append(' = ')
    else:
      strListOperands.append(' + ')
  strOperands = ''.join(strListOperands)
  
  # Results portion
  strListResult = []
  for i in result:
    strListResult.append(i)
  strResult = ''.join(strListResult)
    
  # Decoded operands portion
  strListDecOperands = []
  for i in decodedOperands:
    strListDecOperands.append(str(i))
    if i == decodedOperands[len(decodedOperands)-1]:
      strListDecOperands.append(' = ')
    else:
      strListDecOperands.append(' + ')
  strListDecOperands.append(str(decodedResult))
  strDecOperands = ''.join(strListDecOperands)
      
  
  # Print letter mapping
  letterMap = []
  letterMap.append('Letter Mapping: ')
  for i in range(len(bestAns)):
    if '#' not in bestAns[i]:
      letterMap.append(str(bestAns[i]) + ':' + str(i) + ' ')
  
  strLetterMap = ''.join(letterMap)
  
  # Print fitness
  strFitness = 'Fitness: ' + str(calcFit(bestAns))
  
  # Everything together
  everything = []
  everything.append(strOperands)
  everything.append(strResult)
  everything.append('\n')
  everything.append(strDecOperands)
  everything.append('\n\n')
  everything.append(strLetterMap)
  everything.append('\n\n')
  everything.append(strFitness)
  
  # Combing everything list into a string
  strEverything = ''.join(everything)
  
  print(strEverything)

#************************* START OF MAIN CODE *****************************************************#

# DIFFERENT VERBAL ARITHMETIC PROBLEMS WITH VARYING NUMBER BASES
# ONLY ONE CAN BE COMMENTED OUT AT A TIME
#B = 10
#operands = [['C', 'O', 'C', 'A'], ['C', 'O', 'L', 'A']]
#result = ['O', 'A', 'S', 'I', 'S']

#B = 11
#operands = [['S', 'I', 'L', 'E', 'N', 'C', 'E'], ['I', 'G', 'N', 'I', 'T', 'E', 'S']]
#result = ['C', 'O', 'L', 'L', 'E', 'E', 'N']

B = 10
operands = [['S', 'E', 'N', 'D'], ['M', 'O', 'R', 'E']]
result = ['M', 'O', 'N', 'E', 'Y']

#B = 25
#operands = [['B', 'E', 'E', 'F', 'Y'], ['R', 'U', 'B', 'B', 'E', 'R']]
#result = ['Q', 'U', 'E', 'E', 'N', 'S']

#B = 11
#operands = [['L', 'A', 'G', 'E'], ['L', 'I', 'P', 'P', 'E']]
#result = ['S', 'C', 'H', 'O', 'E', 'N']

#B = 11
#operands = [['U', 'N', 'I', 'T', 'E', 'D'], ['S', 'T', 'A', 'T', 'E', 'S']]
#result = ['A', 'M', 'E', 'R', 'I', 'C', 'A']

#B = 12
#operands = [['L', 'A', 'G', 'E'], ['L', 'E', 'M', 'G', 'O']]
#result = ['S', 'C', 'H', 'O', 'E', 'N']


# GENERATE UNIQUE LETTERS AND FIRST LETTERS FROM OPERANDS AND RESULTS
uniqLetters = []
firstLetters = []
setUpProb()

# Data analysis lists
timeHist = []
genBest = []
wiseHist = []
avgHist = []

for run in range(1):
  start_time = time.time()
  # CREATE INITIAL POPULATION
  ppl = []
  createPpl(200)
  
  bestFitness = 10000000000
  
  # GENETIC ALGORITHM
  totalGen = 10000
  for genNum in range(totalGen):
    # EVALUATE FITNESS OF POPULATION
    pplFitness = []
    evalFitness()
    # Calculate average fitness of ppl
    avgFit = sum(pplFitness) / len(pplFitness)
    avgHist.append(avgFit)
    
    #solutionKey = ['O', 'M', 'Y', '#', '#', 'E', 'N', 'D', 'R', 'S']
    
    # CHECK FOR BEST SOLUTION
    if min(pplFitness) < bestFitness:
      bestFitness = min(pplFitness)
      bestSolution = list(ppl[pplFitness.index(min(pplFitness))])  
    
    # CHECK WHETHER OPTIMAL SOLUTION HAS BEEN FOUND
    if min(pplFitness) == 0:
      break
    
    # SELECT PARENTS
    parents = []
    selectParents()   
    
    # CREATE OFFSPRING
    offspring = []
    createOffspring()
    
    # WISDOM OF CROWDS OFFSPRING
    if genNum % 1 == 0:    
      wiseOffspring = []
      wisdomOfCrowds()
      wiseHist.append(calcFit(wiseOffspring))
    
    # REPLACE WORST CHROMOSOMES WITH OFFSPRING
    ppl[pplFitness.index(max(pplFitness))] = list(offspring)
    ppl[pplFitness.index(max(pplFitness))] = list(wiseOffspring)
    
    
    # MUTATE POPULATION BASED ON MUTATION RATE
    mutateRate = 0.01
    for m in range(int(len(ppl) * mutateRate)):  
      mutate()
    
    # Track number of generations completed by printing to console
    if genNum % 100 == 0:
      print(genNum, min(pplFitness))
    
    # Add best generation value to genBest
    genBest.append(min(pplFitness))
  
  timeHist.append((time.time() - start_time))  


# DISPLAY BEST SOLUTION AS TEXT WIDGET GUI
displaySolution(bestSolution)
  
# DISPLAY PERFORMANCE GRAPH
#plt.figure(1)
#plt.plot(genBest, '-.ro')
#plt.xlabel('Generations')
#plt.ylabel('Cost')
#plt.title('Performance Graph for genBest') 
#
#
#plt.figure(2)
#plt.plot(genBest, '-.ro')
#plt.plot(avgHist, '-.bo')
#plt.xlabel('Generations')
#plt.ylabel('Cost')
#plt.title('Performance Graph for genBest-ro vs avgHist-bo') 
#
#
#plt.figure(3)
#plt.plot(wiseHist, '-.ro')
#plt.plot(avgHist, '-.bo')
#plt.xlabel('Generations')
#plt.ylabel('Cost')
#plt.title('Performance Graph for wiseHist-ro vs avgHist-bo') 
#plt.show()    
#      
#
#plt.figure(4)
#plt.plot(wiseHist, '-.ro')
#plt.plot(genBest, '-.bo')
#plt.xlabel('Generations')
#plt.ylabel('Cost')
#plt.title('Performance Graph for wiseHist-ro vs genBest-bo') 
#plt.show()  







