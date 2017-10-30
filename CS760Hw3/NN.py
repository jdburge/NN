import numpy as np
from collections import defaultdict, Counter, OrderedDict
import math
import copy
import itertools
import operator


class NeuralNetwork(object):
    def __init__(self, learning_rate):
        self.w1 = None
        self.w2 = None
        self.learning_rate = learning_rate

    def train(self, data, meta, target_feature, epochs):
        # Get a list of the features of the data.
        _features = list(data.dtype.names)

        # Make sure valid targetFeature is given.
        if target_feature not in _features:
            raise Exception("Target feature '" + str(target_feature) + "' not found in data. Add target feature name as an additional argument please.")

        # Remove targetFeature from features.
        _features.remove(target_feature)

        # Randomly initialize weights.
        np.random.seed(0)
        # add a +1 for a bias term
        _num_feats = len(_features) + 1
        # Initialize weights in (-1, 1) in matrix of correct size.
        self.w1 = 2 * np.random.random((_num_feats, _num_feats)) - 1
        self.w2 = 2 * np.random.random((_num_feats, 1)) - 1

        _training_input = []
        _training_output = []
        # Assumes target feature is last feature in data.
        for i in data:
            _training_input.append(i.tolist()[:-1] + (1,))
            _training_output.append(i.tolist()[-1])

        _target_feature_on = meta[target_feature][1][0]
        _training_output = [i == _target_feature_on for i in _training_output]
        _training_output = [0 if i else 1 for i in _training_output]

        # Train Model.
        for i in range(epochs):
            # Pass data through network.
            _z = self.sigmoid(np.dot(_training_input, self.w1))
            _output = self.sigmoid(np.dot(_z, self.w2)).flatten()

            # Find error.
            # delta E_total / delta out 2.
            _dE_do2 = np.subtract(_output, _training_output)
            # delta out2 / delta in 2.
            _do2_di2 = self.sigmoid_prime(_output)
            _error_output = np.multiply(_dE_do2, _do2_di2)
            # delta E_total / delta out 2.
            _dE_do1 = np.dot(np.transpose([list(_error_output)]), np.transpose(self.w2))
            # delta out1 / delta net 1.
            _do1_dn1 = self.sigmoid_prime(_z)
            _error_hidden = np.multiply(_dE_do1, _do1_dn1)

            # delta E / delta w2.
            _dE_dw2 = np.dot(np.transpose(_z), np.transpose([list(_error_output)]))
            # delta E / delta w1.
            _dE_dw1 = np.dot(np.transpose(_training_input), _error_hidden)

            self.w2 -= np.multiply(_dE_dw2, self.learning_rate)
            self.w1 -= np.multiply(_dE_dw1, self.learning_rate)

    def test(self, test_data, meta, target_feature, fold, cutoff=0.5, printing=True, ROC=False):
        _test_input = []
        _test_output = []
        # Assumes target feature is last feature in data.
        for i in test_data:
            _test_input.append(i.tolist()[:-1] + (1,))
            _test_output.append(i.tolist()[-1])

        _target_feature_on = meta[target_feature][1][0]
        _test_output = [i == _target_feature_on for i in _test_output]
        _test_output = [0 if i else 1 for i in _test_output]

        # Compute output results and confidence.
        _output = self.forward_pass(_test_input)
        _output_clean = [0 if i < cutoff else 1 for i in _output]
        _output_diff = _test_output - _output.flatten()
        _output_confidence = 1 - np.abs(_output_diff)

        # Print results
        if printing:
            for i in range(len(test_data)):
                print(str(fold) + " " + str(_output_clean[i]) + " " + str(_test_output[i]) + " " + str(_output_confidence[i]))

        if not ROC:
            # Return number of correct predictions
            return np.sum(_output_clean[i] == _test_output[i] for i in range(len(test_data)))
        else:
            _true_positives = 0
            _false_positives = 0
            _true_negatives = 0
            _false_negatives = 0
            for i in range(len(test_data)):
                if _output_clean[i]:
                    if _test_output[i]:
                        _true_positives += 1
                    else:
                        _false_positives += 1
                else:
                    if _test_output[i]:
                        _false_negatives += 1
                    else:
                        _true_negatives += 1
            return _true_positives, _false_positives, _true_negatives, _false_negatives

    def forward_pass(self, input_data):
        _z = self.sigmoid(np.dot(input_data, self.w1))
        _output = self.sigmoid(np.dot(_z, self.w2))
        return _output

    def print_model(self):
        print("w1")
        #for i in self.w1:
        #    print(i)
        print("w2")
        print(self.w2)


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
