# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:33:48 2021

@author: amill
"""

## stochastic hill climbing to optimize a multilayer perceptron for classification
from math import exp
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time 
import matplotlib.pyplot as plt 
import numpy as np 
from collections import Counter 
 
# transfer function
def transfer(activation):
	# sigmoid transfer function
	return 1.0 / (1.0 + exp(-activation))
 
# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation
 
# activation function for a network
def predict_row(row, network):
	inputs = row
	# enumerate the layers in the network from input to output
	for layer in network:
		new_inputs = list()
		# enumerate nodes in the layer
		for node in layer:
			# activate the node
			activation = activate(inputs, node)
			# transfer activation
			output = transfer(activation)
			# store output
			new_inputs.append(output)
		# output from this layer is input to the next layer
		inputs = new_inputs
	return inputs[0]
 
# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, network):
	yhats = list()
	for row in X:
		yhat = predict_row(row, network)
		yhats.append(yhat)
	return yhats
 
# objective function
def objective(X, y, network):
	# generate predictions for dataset
	yhat = predict_dataset(X, network)
	# round the predictions
	yhat = [round(y) for y in yhat]
	# calculate accuracy
	score = accuracy_score(y, yhat)
	return score
 
# take a step in the search space
def step(network, step_size):
	new_net = list()
	# enumerate layers in the network
	for layer in network:
		new_layer = list()
		# enumerate nodes in this layer
		for node in layer:
			# mutate the node
			new_node = node.copy() + randn(len(node)) * step_size
			# store node in layer
			new_layer.append(new_node)
		# store layer in network
		new_net.append(new_layer)
	return new_net


def hillclimbing(X, y, objective, solution, n_iter, step_size):
    #evaluate the initial point 
    solution_eval = objective(X, y, solution)
    #run the hill climb 
    step_size_when_energy_change = []
    schedule = [1, 0.7, 0.5, 0.3, 0.2, 0.1]
    for i in schedule:
        step_size = i
        for i in range(round(1000/len(schedule))):
            #take a step 
            candidate = step(solution, step_size)
            #evaluate candidate point 
            candidte_eval = objective(X, y, candidate)
            #check if we should the keep the new point 
            if candidte_eval >= solution_eval:
                #adding the step_size to the histogram list 
                step_size_when_energy_change.append(round(step_size, 2))
                #store the new point 
                solution, solution_eval = candidate, candidte_eval
                #report progress 
                print('>%d %f' % (i, solution_eval))
        if step_size < 0.01:
            break 
        print(step_size)
    return [solution, solution_eval, step_size_when_energy_change]

samples = 1000
scoring = []
times = []
accepted_moves_hist = []
for i in range(samples):
    start = time.time()
    # define dataset
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # define the total iterations
    n_iter = 100
    # define the maximum step size
    step_size = 1
    # determine the number of inputs
    n_inputs = X.shape[1]
    # one hidden layer and an output layer
    n_hidden = 10
    hidden1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
    output1 = [rand(n_hidden + 1)]
    network = [hidden1, output1]
    # perform the hill climbing search
    network, score, hist = hillclimbing(X_train, y_train, objective, network, n_iter, step_size)
    print('Done!')
    print('Best: %f' % (score))
    accepted_moves_hist.append(hist)
    # generate predictions for the test dataset
    yhat = predict_dataset(X_test, network)
    # round the predictions
    yhat = [round(y) for y in yhat]
    # calculate accuracy
    score = accuracy_score(y_test, yhat)
    scoring.append(score)
    print('Test Accuracy: %.5f' % (score * 100))
    end = time.time()
    times.append(end-start)
    print('Sample ' + str(i+1) + ' Complete')
    
mean_time = np.mean(times)
mean_score = np.mean(scoring)
var_score = np.var(scoring)
print('The mean accuracy is ' + str(mean_score*100) + '%')
print('The variance on the accuracy is ' + str(var_score*100) + '%')
print('The average time taken to compute the model is ' + str(mean_time))

#needed for flattening a list 
def flatten(t):
    return [item for sublist in t for item in sublist]

#creating a histogram of accepted moves 
flattened_hist_vals = flatten(accepted_moves_hist)
hist_vals = Counter(flattened_hist_vals)
keys = hist_vals.keys()
vals = hist_vals.values()

def avg_accepted_moves_per_step_hist(keys, vals):
    plt.bar(keys, (np.divide(list(vals), samples)), width = 0.05)
    plt.ylabel('Average Number of Accepted Moves')
    plt.xlabel('Step Size')
    plt.title('Average Number of Accepted Moves per Step Size\n Averaged over %i Samples \n 0.7 Deflation Schedule' %samples)
    return plt.show()

avg_accepted_moves_per_step_hist(keys, vals)