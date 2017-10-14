from scipy.io import arff
import numpy as np
import ID3
import matplotlib.pyplot as plt
from scipy.interpolate import spline

# Training set sizes.
_sizes = [0.05, 0.1, 0.2, 0.5, 1]
# Stopping criteria.
m = 4

# Load heart data.
data, meta = arff.loadarff('heart_train.arff')
test_data, test_meta = arff.loadarff('heart_test.arff')

# Set up figure for heart data.
fig = plt.figure()

fig.suptitle('Part 2: Learning Curves', fontsize=14, fontweight='bold')
ax = fig.add_subplot(121)
fig.subplots_adjust(top=0.85)
ax.set_title('Heart Data')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('Accuracy on Test Set')

# Collect averages of different set sizes to use to compute a learning curve.
_averages = []

# Iterate through the different set sizes.
for _size in _sizes:
    # For all sizes but the whole set, iterate 10 times on random data.
    if _size != 1:
        _scores = []
        for i in range(0, 10):
            _rand_data = np.random.choice(data, int(round(_size * len(data))), replace=False, )
            myID3 = ID3.ID3Classifier(m)
            myID3.train(_rand_data, meta, 'class')
            _scores.append(myID3.test(test_data, 'class'))
        _average_score = sum(_scores) / float(len(_scores))
        _max_score = max(_scores)
        _min_score = min(_scores)
        _averages.append(_average_score)
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 1)))
        c = next(color)
        ax.plot([_size, _size, _size], [_average_score, _max_score, _min_score], 'o', label=_size)
    else:
        myID3 = ID3.ID3Classifier(m)
        myID3.train(data, meta, 'class')
        _score = myID3.test(test_data, 'class')
        _averages.append(_score)
        ax.plot([_size], [_score], 'o', label=_size)

# Compute learning curve.
_sizes_np = np.array(_sizes)
_averages_np = np.array(_averages)
_sizes_new = np.linspace(_sizes_np.min(), _sizes_np.max(), 300)
_averages_smooth = spline(_sizes_np, _averages_np, _sizes_new)
ax.plot(_sizes_new, _averages_smooth)

ax.axis([0, 1, 0, 1])

# Execute the same code on the diabetes data.
data, meta = arff.loadarff('diabetes_train.arff')
test_data, test_meta = arff.loadarff('diabetes_test.arff')

ax2 = fig.add_subplot(122)
fig.subplots_adjust(top=0.85)
ax2.set_title('Diabetes Data')
ax2.set_xlabel('Training Set Size')
ax2.set_ylabel('Accuracy on Test Set')

_averages = []

for _size in _sizes:
    if _size != 1:
        _scores = []
        for i in range(0, 10):
            _rand_data = np.random.choice(data, int(round(_size * len(data))), replace=False, )
            myID3 = ID3.ID3Classifier(m)
            myID3.train(_rand_data, meta, 'class')
            _scores.append(myID3.test(test_data, 'class'))
        _average_score = sum(_scores) / float(len(_scores))
        _max_score = max(_scores)
        _min_score = min(_scores)
        _averages.append(_average_score)
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 1)))
        c = next(color)
        ax2.plot([_size, _size, _size], [_average_score, _max_score, _min_score], 'o', label=_size)
    else:
        myID3 = ID3.ID3Classifier(m)
        myID3.train(data, meta, 'class')
        _score = myID3.test(test_data, 'class')
        _averages.append(_score)
        ax2.plot([_size], [_score], 'o', label=_size)

_sizes_np = np.array(_sizes)
_averages_np = np.array(_averages)
_sizes_new = np.linspace(_sizes_np.min(), _sizes_np.max(), 300)
_averages_smooth = spline(_sizes_np, _averages_np, _sizes_new)
ax2.plot(_sizes_new, _averages_smooth)

ax2.axis([0, 1, 0, 1])

# Plot.
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
