from scipy.io import arff
import numpy as np
import ID3
import matplotlib.pyplot as plt
from scipy.interpolate import spline

# Stopping criteria list.
_ms = [2, 5, 10, 20]

# Load heart data.
data, meta = arff.loadarff('heart_train.arff')
test_data, test_meta = arff.loadarff('heart_test.arff')

# Set up figure.
fig = plt.figure()

fig.suptitle('Part 3: Tree Sizes', fontsize=14, fontweight='bold')
ax = fig.add_subplot(121)
fig.subplots_adjust(top=0.85)
ax.set_title('Heart Data')
ax.set_xlabel('Stopping Criteria Value')
ax.set_ylabel('Accuracy on Test Set')

# Hold scores of different stopping criteria to create curve at end.
_scores = []

# Get scores at different stopping criteria.
for _m in _ms:
    _score = 0
    myID3 = ID3.ID3Classifier(_m)
    myID3.train(data, meta, 'class')
    _score = myID3.test(test_data, 'class')
    _scores.append(_score)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 1)))
    c = next(color)
    ax.plot([_m], [_score], 'o', label=_m)

# Plot curve.
_ms_np = np.array(_ms)
_scores_np = np.array(_scores)
_ms_new = np.linspace(_ms_np.min(), _ms_np.max(), 300)
_scores_smooth = spline(_ms_np, _scores_np, _ms_new)
ax.plot(_ms_new, _scores_smooth)

ax.axis([0, 20, 0, 1])

# Load diabetes data.
data, meta = arff.loadarff('diabetes_train.arff')
test_data, test_meta = arff.loadarff('diabetes_test.arff')

ax2 = fig.add_subplot(122)
fig.subplots_adjust(top=0.85)
ax2.set_title('Diabetes Data')
ax2.set_xlabel('Stopping Criteria Value')
ax2.set_ylabel('Accuracy on Test Set')

_scores = []

for _m in _ms:
    _score = 0
    myID3 = ID3.ID3Classifier(_m)
    myID3.train(data, meta, 'class')
    _score = myID3.test(test_data, 'class')
    _scores.append(_score)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 1)))
    c = next(color)
    ax2.plot([_m], [_score], 'o', label=_m)

_ms_np = np.array(_ms)
_scores_np = np.array(_scores)
_ms_new = np.linspace(_ms_np.min(), _ms_np.max(), 300)
_scores_smooth = spline(_ms_np, _scores_np, _ms_new)
ax2.plot(_ms_new, _scores_smooth)

ax2.axis([0, 20, 0, 1])

# Plot.
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
