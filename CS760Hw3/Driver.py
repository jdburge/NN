from scipy.io import arff
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import NN

data, meta = arff.loadarff('sonar.arff')

np.random.shuffle(data)

size = len(data)
data_sets = dict()
train_sets = dict()
for i in range(10):
    data_sets[i] = []
    train_sets[i] = []

i = 0
for x in data:
    data_sets[int((10 * i)/size)].append(x)
    i += 1

for i in range(10):
    data_sets[i] = np.array(data_sets[i])

for i in range(10):
    for j in range(10):
        if i != j:
            for x in data_sets[j]:
                train_sets[i].append(x)
    train_sets[i] = np.array(train_sets[i])

_correct_predictions = 0
for i in range(10):
    # Give neural network the learning rate.
    # NOTE: Data must have a target feature 'Class' in the last indice.
    myNN = NN.NeuralNetwork(0.1)
    myNN.train(train_sets[i], meta, 'Class', 100)
    _correct_predictions += myNN.test(data_sets[i], meta, 'Class', i)

print("Correct: " + str(_correct_predictions) + " Incorrect: " + str(size - _correct_predictions))
