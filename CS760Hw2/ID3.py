import numpy as np
from collections import defaultdict, Counter
import math
import copy
import itertools


class ID3Classifier(object):
    def __init__(self, m):
        self.root = None
        self.m = m

    def train(self, data, meta, target_feature):
        # Get a list of the features of the data.
        _features = list(data.dtype.names)

        # Make sure valid targetFeature is given.
        if target_feature not in _features:
            raise Exception("Target feature not found in data.")

        # Remove targetFeature from features.
        _features.remove(target_feature)

        # Train Model.
        self.root = self.create_tree(data, target_feature, _features, meta, data[0][target_feature].decode('UTF-8'))

    def create_tree(self, data, target_feature, features, meta, parent_common):
        # Create a root node for the tree.
        root = None

        # If data is empty, return parent common.
        if len(data) == 0:
            root = LeafNode(parent_common, self.print_target(data, target_feature))
            return root
        # If all examples are the same, return root with correct label.
        _ret = self.are_values_same(data, target_feature)
        if _ret is not None:
            root = LeafNode(_ret, self.print_target(data, target_feature))
        # If number of predicting attributes is empty, return root with a label of the most common value in the data.
        elif not features:
            _most_common = self.find_most_common(data, target_feature, parent_common)
            root = LeafNode(_most_common, self.print_target(data, target_feature))
        # Create tree.
        else:
            # Get target_feature.
            _values = data[target_feature]
            # If the number of data points is less than m, return root with label for most common value.
            if len(_values) < self.m:
                _most_common = self.find_most_common(data, target_feature, parent_common)
                root = LeafNode(_most_common, self.print_target(data, target_feature))
            else:
                # Find the feature that best classifies.
                _new_features = copy.copy(features)
                _best_feature = self.find_best_feature(data, _new_features, _values, target_feature, meta)
                # Set feature to the root.
                root = InternalNode(_best_feature, self.print_target(data, target_feature))
                # Determine all values for this feature.
                _feature_values = [i[_best_feature] for i in data]
                # If the feature is numeric.
                if type(_feature_values[0]) is np.float64:
                    # Find midpoint.
                    _midpoint = self.compute_midpoint(data, _best_feature, target_feature)
                    # Leaf for <= midpoint.
                    _less_than_data = np.array([i for i in data if i[_best_feature] <= _midpoint])
                    _parent_common = self.find_most_common(data, target_feature, parent_common)
                    _child_less = self.create_tree(_less_than_data, target_feature, _new_features, meta, _parent_common)
                    _key = '<='
                    root[_key] = _child_less
                    # Leaf for >= midpoint.
                    _greater_than_data = np.array([i for i in data if i[_best_feature] > _midpoint])
                    _parent_common = self.find_most_common(data, target_feature, parent_common)
                    _child_greater = self.create_tree(_greater_than_data, target_feature, _new_features, meta, _parent_common)
                    _key = '>'
                    root[_key] = _child_greater
                    root.midpoint = _midpoint
                # If the feature is categorical.
                else:
                    for _value in meta[_best_feature][1]:
                        # Create a subset of examples that have this value.
                        _value_data = np.array([i for i in data if i[_best_feature].decode('UTF-8') == _value])
                        # Add a new tree branch below the root for the value.
                        # If the subset is empty, set leaf node label to the most common value in the examples.
                        if _value_data is None:
                            _most_common = self.find_most_common(data, target_feature)
                            _child = LeafNode(_most_common)
                        # Otherwise recurse on ID3.
                        else:
                            _parent_common = self.find_most_common(data, target_feature, parent_common)
                            _child = self.create_tree(_value_data, target_feature, _new_features, meta, _parent_common)
                        # Set child on this path of the tree.
                        _key = str(_value)
                        root[_key] = _child

        # Ensure that root was set.
        if root is None:
            raise Exception("Root was not set to a valid node.")

        # Return root.
        return root

    def print_tree(self):
        self.print_node(self.root, 0)

    def print_node(self, node, depth):
        if not node.is_leaf():
            for child in node:
                print("")
                for i in range(0, depth):
                    print "|\t",
                print str(node),
                if node.midpoint is None:
                    print "=",
                print str(child),
                if node.midpoint is not None:
                    print str(node.midpoint),
                print node[child].str_targets(),
                self.print_node(node[child], depth + 1)
        elif node.is_leaf():
            print str(node),

    @staticmethod
    def print_target(data, target_feature):
        if len(data) == 0:
            return "[0 0]"
        _values, _counts = np.unique(data[target_feature], return_inverse=True)
        _bincount = np.bincount(_counts)
        if len(_bincount) == 1:
            if _values[0].decode('UTF-8') == 'negative':
                return "[" + str(_bincount[0]) + " 0]"
            elif _values[0].decode('UTF-8') == 'positive':
                return "[0 " + str(_bincount[0]) + "]"
        return str(_bincount)

    @staticmethod
    # Return the most common common label value.
    def find_most_common(data, target_feature, parent_common):
        _values, _counts = np.unique(data[target_feature], return_inverse=True)
        _bincount = np.bincount(_counts)
        # If same amount of both values, return parent common.
        if len(_bincount) == 2 and _bincount[0] == _bincount[1]:
            return parent_common
        return _values[np.argmax(_bincount)].decode('UTF-8')

    @staticmethod
    # Determine if all values in the data are the same for the target_feature.
    def are_values_same(data, target_feature):
        _size = len(data)
        _values, _counts = np.unique(data[target_feature], return_inverse=True)
        for i in range(0, len(_values)):
            if np.bincount(_counts)[i] == _size:
                return _values[0].decode('UTF-8')
        return None

    # Determine the feature that gives the best information gain.
    def find_best_feature(self, data, features, values, target_feature, meta):
        _gain_amount = [(self.calculate_info_gain(data, feature, values, target_feature), feature) for feature in features]
        # Note this sorting maintains ordering of the original ARFF file in case of ties.
        _gain_amount.sort(key=lambda x: x[0])
        # Maintain ordering of the features.
        i = 1
        while i < len(_gain_amount) and _gain_amount[-1-i][0] == _gain_amount[-1][0]:
            i += 1
        _best_feature = _gain_amount[-i][1]
        # Remove feature used this iteration of ID3 if feature is not numeric.
        if meta[_best_feature][0] != 'numeric':
            features.pop(features.index(_best_feature))
        return _best_feature

    # Calculate the information gain for the given feature.
    def calculate_info_gain(self, data, feature, values, target_feature):

        _size = len(data)
        _data_sets = defaultdict(list)
        _feature_entropy = 0

        for i in data:
            _data_sets[i[feature]].append(i[target_feature])

        if type(data[0][feature]) is np.float64:
            _feature_entropy = self.compute_best_split(_data_sets, data)
        else:
            for _set in _data_sets.values():
                _feature_entropy += (float(len(_set)) / float(_size)) * float(self.entropy(_set))

        return self.entropy(values) - _feature_entropy

    @staticmethod
    # Calculate the entropy on the data.
    def entropy(data):
        _size = len(data)
        _counter = Counter(data)
        return sum(-1*(float(_counter[i]) / float(_size)) * math.log((float(_counter[i]) / float(_size)), 2)
                  for i in _counter)

    # Compute best splitting threshold.
    def compute_best_split(self, data_sets, data):
        _nums = [i for i in data_sets]
        _nums.sort()
        _lowest_feature_entropy = 1
        # Split at value _nums[i].
        for i in range(1, len(_nums)):
            _lower = [_nums[k] for k in range(0, i)]
            _upper = [_nums[k] for k in range(i, len(_nums))]

            _lower_set = list(itertools.chain.from_iterable([data_sets[k] for k in data_sets if k in _lower]))
            _upper_set = list(itertools.chain.from_iterable([data_sets[k] for k in data_sets if k in _upper]))

            _feature_entropy = (len(_lower_set) / float(len(data))) * self.entropy(_lower_set) + \
                               (len(_upper_set) / float(len(data))) * self.entropy(_upper_set)

            if _feature_entropy < _lowest_feature_entropy:
                _lowest_feature_entropy = _feature_entropy

        return _lowest_feature_entropy

    # Determine midpoint.
    def compute_midpoint(self, data, feature, target_feature):
        _data_sets = defaultdict(list)
        for i in data:
            _data_sets[i[feature]].append(i[target_feature])

        _nums = [i for i in _data_sets]
        _nums.sort()

        _lowest_feature_entropy = 1
        _midpoint = 0
        # Split at value _nums[i].
        for i in range(0, len(_nums)):
            _lower = [_nums[k] for k in range(0, i)]
            _upper = [_nums[k] for k in range(i, len(_nums))]

            _lower_set = list(itertools.chain.from_iterable([_data_sets[k] for k in _data_sets if k in _lower]))
            _upper_set = list(itertools.chain.from_iterable([_data_sets[k] for k in _data_sets if k in _upper]))

            _feature_entropy = (len(_lower_set) / float(len(data))) * self.entropy(_lower_set) + \
                               (len(_upper_set) / float(len(data))) * self.entropy(_upper_set)

            if _feature_entropy < _lowest_feature_entropy:
                _lowest_feature_entropy = _feature_entropy
                if i != 0:
                    _midpoint = (_nums[i] + _nums[i - 1]) / 2
                else:
                    _midpoint = _nums[i]

        return _midpoint

    def predict(self, test_set):
        _classifications = []

        for i in test_set:
            _class = None
            _node = self.root
            while _class is None:
                if _node.is_leaf():
                    _class = _node.value
                else:
                    _feature_value = i[_node.feature]
                    # Numeric node.
                    if _node.midpoint is not None:
                        if _feature_value <= _node.midpoint:
                            _node = _node['<=']
                        elif _feature_value > _node.midpoint:
                            _node = _node['>']
                    else:
                        _node = _node[_feature_value.decode('UTF-8')]
            _classifications.append(_class)

        return _classifications

    def test(self, test_data, target_feature, printing=False):
        if printing:
            print("\n<Predictions for the Test Set Instances>")
        # Get a list of the features of the data.
        _features = list(test_data.dtype.names)

        # Make sure valid targetFeature is given.
        if target_feature not in _features:
            raise Exception("Target feature not found in data.")

        # Remove targetFeature from features.
        _features.remove(target_feature)

        targets = [i.decode('UTF-8') for i in test_data[:][target_feature]]

        _classifications = self.predict(test_data)
        correct = 0
        for i in range(0, len(targets)):
            if printing:
                print(str(i) + ": Actual: " + str(_classifications[i]) + " Predicted: " + str(targets[i]))
            correct += _classifications[i] == targets[i]
        if printing:
            print("Number of correctly classified: " + str(correct) + " Total number of test instances: " + str(len(targets)))

        return correct / len(test_data)


# Class for handling internal nodes in the decision tree.
class InternalNode(dict):
    def __init__(self, feature, targets,  *args, **kwargs):
        self.feature = feature
        self.targets = targets
        self.midpoint = None
        super(InternalNode, self).__init__(*args, **kwargs)

    def is_leaf(self):
        return False

    def str_targets(self):
        return "%s" % self.targets

    def __repr__(self):
        return "%s" % self.feature


# Class for handling leaf nodes in the decision tree.
class LeafNode(dict):
    def __init__(self, value, targets, *args, **kwargs):
        self.value = value
        self.targets = targets
        super(LeafNode, self).__init__(*args, **kwargs)

    def is_leaf(self):
        return True

    def str_targets(self):
        return "%s" % self.targets

    def __repr__(self):
        return ": %s" % self.value
