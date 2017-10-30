from scipy.io import arff
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import NN
import matplotlib.pyplot as plt
from scipy.interpolate import spline

data, meta = arff.loadarff('sonar.arff')

# Epoch sizes
_epochs = [25, 50, 75, 100]
_learning_rate = 0.1
_folds = 10
_target = 'Class'

# Set up figure.
fig = plt.figure()

fig.suptitle('Part B: Epochs', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Accuracy on Epoch Sizes')
ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Accuracy on Test Set')

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

for epoch in _epochs:
    _correct_predictions = 0
    for i in range(_folds):
        # Give neural network the learning rate.
        # NOTE: Data must have a target feature 'Class' in the last indice.
        myNN = NN.NeuralNetwork(_learning_rate)
        myNN.train(train_sets[i], meta, _target, epoch)
        _correct_predictions += myNN.test(data_sets[i], meta, _target, i, printing=False)

    _accuracy = _correct_predictions / float(size)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 1)))
    c = next(color)
    ax.plot(epoch, _accuracy, 'o', label=epoch)

ax.axis([0, 100, 0, 1])

# Plot.
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()