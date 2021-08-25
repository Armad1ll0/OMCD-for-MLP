# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:13:36 2021

@author: amill
"""

import numpy as np
import matplotlib.pyplot as plt 
from numpy import asarray
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
import sympy as sym 

#objective function 
def objective(z):
    return np.tanh(z)*np.sin(z)*(np.cos(z))**2

def hill_climb(objective, bounds, n_iterations, step_size):
    #generate an initial point 
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # evaluate the initial point
    solution_eval = objective(solution)
    #run the hill climb 
    scores = list()
    scores.append(solution_eval)
    count = 0
    while step_size > 0:
        for i in range(n_iterations):
            #taking a step 
            candidate = solution + randn(len(bounds))*step_size
            #evaluating the candidate point 
            candidate_eval = objective(candidate)
            #checking if it is worth keeping
            count += 1
            if candidate_eval <= solution_eval:
                #store the new point 
                solution, solution_eval = candidate, candidate_eval 
                #keeping track of scores 
                scores.append(solution_eval)
                #letting us know about the progress 
                print('>%d f(%s) = %.5f' %(count, solution, solution_eval))
        step_size = step_size - 0.1
    return [solution, solution_eval, scores]

#plotting the objective function 
z = np.linspace(-10000,10000,100)
y = np.tanh(z)*np.sin(z)*(np.cos(z))**2
# plot the function
plt.plot(z,y, 'r')
# show the plot
plt.show()

# seed the pseudorandom number generator
seed(5)
# define range for input
bounds = asarray([[-10.0, 10.0]])
# define the total iterations
n_iterations = 100
# define the maximum step size
step_size = 100
# perform the hill climbing search
best, score, scores = hill_climb(objective, bounds, n_iterations, step_size)
print('Done!')
print('f(%s) = %f' % (best, score))

#performing gradient descent to make sure we have the bottom of the hill 
cur_x = best # The algorithm starts at x=3
rate = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
t = sym.Symbol('t')
#need to fnd a function that can differentiate our function given 
#NEED TO FIGURE OUT HOW TO MAKE THS ALL SEEMLESS AS OPPOSED TO HAVING IT ALL DONE SEPERATELY 
diff = sym.diff(sym.tanh(t)*sym.sin(t)*(sym.cos(t))**2)
#Gradient of our function
df = lambda x: (1 - np.tanh(x)**2)*np.sin(x)*np.cos(x)**2 - 2*np.sin(x)**2*np.cos(x)*np.tanh(x) + np.cos(x)**3*np.tanh(x) #Gradient of our function

#gradient descent function 
while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x #Store current x value in prev_x
    cur_x = cur_x - rate * df(prev_x) #Grad descent
    previous_step_size = abs(cur_x - prev_x) #Change in x
    iters = iters+1 #iteration count
    #print("Iteration",iters,"\nX value is",cur_x) #Print iterations
    
print("The local minimum occurs at", cur_x)
print('and that value is at ' + str(objective(cur_x)))
# line plot of best scores
plt.plot(scores, '.-')
plt.xlabel('Improvement Number')
plt.ylabel('Evaluation f(x)')
plt.show()