from scipy.io import arff
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import NN
import matplotlib.pyplot as plt
from scipy.interpolate import spline

data, meta = arff.loadarff('sonar.arff')

# Epoch sizes
_epochs = 50
_learning_rate = 0.1
_folds = 10
_target = 'Class'
_cutoffs = [0.05, 0.15, 0.25, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70,
            0.75, 0.85, 0.95]


# Set up figure.
fig = plt.figure()

fig.suptitle('Part B: ROC', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('ROC Curve')
ax.set_xlabel('1 - Specificity (false positive rate)')
ax.set_ylabel('Sensitivity (true positive rate')

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


_true_positives = dict()
_false_positives = dict()
_true_negatives = dict()
_false_negatives = dict()
for cutoff in _cutoffs:
    _true_positives[cutoff] = 0
    _false_positives[cutoff] = 0
    _true_negatives[cutoff] = 0
    _false_negatives[cutoff] = 0

for i in range(_folds):
    # Give neural network the learning rate.
    # NOTE: Data must have a target feature 'Class' in the last indice.
    myNN = NN.NeuralNetwork(_learning_rate)
    myNN.train(train_sets[i], meta, _target, _epochs)
    for cutoff in _cutoffs:
        _true_positive, _false_positive, _true_negative, _false_negative = \
            myNN.test(data_sets[i], meta, _target, i, cutoff, printing=False, ROC=True)
        _true_positives[cutoff] += _true_positive
        _false_positives[cutoff] += _false_positive
        _true_negatives[cutoff] += _true_negative
        _false_negatives[cutoff] += _false_negative

for cutoff in _cutoffs:
    x = _false_positives[cutoff]/float(_false_positives[cutoff] + _true_negatives[cutoff] + 0.00001)
    y = _true_positives[cutoff]/float(_true_positives[cutoff] + _false_negatives[cutoff] + 0.00001)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 1)))
    c = next(color)
    ax.plot(x, y, 'o', label=cutoff)

ax.axis([0, 1, 0, 1])

# Plot.
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()