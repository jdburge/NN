import sys
from scipy.io import arff
import ID3

arg1 = str(sys.argv[1])
arg2 = str(sys.argv[2])

data, meta = arff.loadarff(arg1)
test_data, test_meta = arff.loadarff(arg2)

m = int(sys.argv[3])

myID3 = ID3.ID3Classifier(m)

myID3.train(data, meta, 'class')
myID3.print_tree()
myID3.test(test_data, 'class', printing=True)
