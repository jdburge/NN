import sys
from scipy.io import arff
import numpy as np
import NN

_file = str(sys.argv[1])
_folds = int(str(sys.argv[2]))
_learning_rate = float(str(sys.argv[3]))
_epochs = int(str(sys.argv[4]))
if len(sys.argv) == 6:
    _target = str(sys.argv[5])
else:
    _target = 'Class'

data, meta = arff.loadarff(_file)

if len(sys.argv) != 6:
    _target = meta.names()[-1]

np.random.shuffle(data)

size = len(data)
data_sets = dict()
train_sets = dict()
for i in range(_folds):
    data_sets[i] = []
    train_sets[i] = []

i = 0
for x in data:
    data_sets[int((_folds * i)/size)].append(x)
    i += 1

for i in range(_folds):
    data_sets[i] = np.array(data_sets[i])

for i in range(_folds):
    for j in range(_folds):
        if i != j:
            for x in data_sets[j]:
                train_sets[i].append(x)
    train_sets[i] = np.array(train_sets[i])

correct_predictions = 0
for i in range(_folds):
    # Give neural network the learning rate.
    # NOTE: Data must have a target feature 'Class' in the last indice.
    myNN = NN.NeuralNetwork(_learning_rate)
    myNN.train(train_sets[i], meta, _target, _epochs)
    correct_predictions += myNN.test(data_sets[i], meta, _target, i)
