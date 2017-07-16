"""
#### Neural Network Description #######
# Network: Multi Layer Neural Network
# Input : Feature vectors from the image.
# Output : A confusion matrix which states the the orientation of the image and the accuracy of the system.
# This is a network with 1 input layer network, 1 hidden layer and one output layer.
# The 1st is the input layer with 192 nodes.  The input to this layer i the values from the feature vector.
# The Second layer is the hidden layer with the number of nodes as given in the input and the input to these nodes, are the output of the previous layer
# The third is the output layer with 4 nodes or classes(i.e 0,90,180,270).
# the parameters for learning_rate= 0.5 and  epoc= 2 ( we have chosen a lower value for epoc to reduce the number of iterations as out training data is huge)
# Activation function used : sigmoid function
#  we have used stocastic gradient descent to update te weights of the network.
"""

# Reference : http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
from random import seed
from random import random
from math import exp
import copy

# confusion matrix
def confusion_mat(con_dict, accuracy):

    print "CONFUSION MATRIX"
    print "------------------------------------------------------------------"
    print "\t\t" + "0\t\t" + "90\t\t" + "180\t\t" + "270\t\t"
    print "0\t\t" + str(con_dict['0'][0]) + "\t\t" + str(con_dict['0'][1]) + "\t\t" + str(con_dict['0'][2]) + "\t\t" + str(con_dict['0'][3])
    print "90\t\t" + str(con_dict['90'][0]) + "\t\t" + str(con_dict['90'][1]) + "\t\t" + str(con_dict['90'][2]) + "\t\t" + str(con_dict['90'][3])
    print "180\t\t" + str(con_dict['180'][0]) + "\t\t" + str(con_dict['180'][1]) + "\t\t" + str(con_dict['180'][2]) + "\t\t" + str(con_dict['180'][3])
    print "270\t\t" + str(con_dict['270'][0]) +  "\t\t" +str(con_dict['270'][1]) +  "\t\t" +str(con_dict['270'][2]) +  "\t\t" +str(con_dict['270'][3])
    print "------------------------------------------------------------------"
    print "Accuracy = " + str(accuracy) + "%"

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * int(inputs[i])
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    if len(row) != 192:
        row.pop(0)
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

seed(1)


def read_features(train_file):
    file = open(train_file, 'r')
    feature_vector = {}
    i = 0
    for line in file:
        if i == 4:
            i = 0
        list_features = []
        list_features += [feature for feature in line.split()]
        feature_vector[list_features[0] + str(i)] = []
        feature_vector[list_features[0] + str(i)] += list_features[1:]
        i += 1

    return feature_vector


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
    return network

# Update network weights with error
def update_weights(network, row, l_rate):
    if len(row) != 192:
        row.pop(0)
    for i in range(len(network)):
		inputs = row
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)-1):
				neuron['weights'][j] += l_rate * neuron['delta'] * int(inputs[j])
			neuron['weights'][-1] += l_rate * neuron['delta']
    return network

# train network
def train_network(network, train, l_rate, n_epoch):
    copied_train_vector = copy.deepcopy(train)
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, train[row])
            if int(copied_train_vector[row][0]) == 0:
                expected = [1, 0, 0, 0]
            elif int(copied_train_vector[row][0]) == 90:
                expected = [0, 1, 0, 0]
            elif int(copied_train_vector[row][0]) == 180:
                expected = [0, 0, 1, 0]
            else:
                expected = [0, 0, 0, 1]
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            network = backward_propagate_error(network, expected)
            network = update_weights(network, train[row], l_rate)
    return network


def read_test_features(test_file):
    file = open(test_file, 'r')
    feature_vector = {}
    for line in file:
        list_features = []
        list_features += [feature for feature in line.split()]
        feature_vector[list_features[0]] = []
        feature_vector[list_features[0]] += list_features[1:]

    return feature_vector

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

def output_file(adaboost_out):
    file = open("nnet_output.txt", "w")
    for key in adaboost_out:
        file.write(str(key) + " " + str(adaboost_out[key]) + '\n')


def read_data(train_file, test_file, hidden_count):
    print "running nn"
    feature_vector = read_features(train_file)
    test_vector = read_test_features(test_file)
    network = initialize_network(192, hidden_count, 4)
    network = train_network(network, feature_vector, 0.5, 4)

    correct_classified = 0
    confusion_dict = {'0':[0,0,0,0], '90':[0,0,0,0], '180':[0,0,0,0], '270':[0,0,0,0]}
    copied_test_vector = copy.deepcopy(test_vector)
    nnout_file = {}
    for image in test_vector:
        actual_orientation = str(copied_test_vector[image][0])
        prediction = predict(network, test_vector[image])
        if prediction == 0:
            nnout_file[image] = prediction
        elif prediction == 1:
            nnout_file[image] = 90
        elif prediction == 2:
            nnout_file[image] = 180
        else:
            nnout_file[image] = 270

        if prediction == 0 and int(copied_test_vector[image][0]) == 0:
            correct_classified += 1
            confusion_dict[actual_orientation][0] += 1
        elif prediction == 1 and int(copied_test_vector[image][0]) == 90:
            correct_classified += 1
            confusion_dict[actual_orientation][1] += 1
        elif prediction == 2 and int(copied_test_vector[image][0]) == 180:
            correct_classified += 1
            confusion_dict[actual_orientation][2] += 1
        elif prediction == 3 and int(copied_test_vector[image][0]) == 270:
            correct_classified += 1
            confusion_dict[actual_orientation][3] += 1

        if str(prediction) == '0' and actual_orientation == '90':
            confusion_dict[actual_orientation][0] += 1
        elif str(prediction) == '0' and actual_orientation == '180':
            confusion_dict[actual_orientation][0] += 1
        elif str(prediction) == '0' and actual_orientation == '270':
            confusion_dict[actual_orientation][0] += 1
        elif str(prediction) == '1' and actual_orientation == '0':
            confusion_dict[actual_orientation][1] += 1
        elif str(prediction) == '1' and actual_orientation == '180':
            confusion_dict[actual_orientation][1] += 1
        elif str(prediction) == '1' and actual_orientation == '270':
            confusion_dict[actual_orientation][1] += 1
        elif str(prediction) == '2' and actual_orientation == '0':
            confusion_dict[actual_orientation][2] += 1
        elif str(prediction) == '2' and actual_orientation == '90':
            confusion_dict[actual_orientation][2] += 1
        elif str(prediction) == '2' and actual_orientation == '270':
            confusion_dict[actual_orientation][2] += 1
        elif str(prediction) == '3' and actual_orientation == '0':
            confusion_dict[actual_orientation][3] += 1
        elif str(prediction) == '3' and actual_orientation == '90':
            confusion_dict[actual_orientation][3] += 1
        elif str(prediction) == '3' and actual_orientation == '180':
            confusion_dict[actual_orientation][3] += 1

    accuracy = float(correct_classified) / len(test_vector)

    confusion_mat(confusion_dict, accuracy * 100)
    output_file(nnout_file)





