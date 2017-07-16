"""
Adaboost Algorithm

Problem Formulation:
The images need to be classified based on the features given and there are 4 classes in total. Since there are 4 classes
we need to build 4 classifiers which will comprise of an ensemble of decision stumps. Each classifier will vote for
CLASS or NOT CLASS.

Based on the number of decision stumps given as input, those many number of i,j pairs are generated randomly for each
class. The images are calssified by comparing i,j columns of each image (i > j). The error and the alpha(hypothesis
weight) are calculated and the misclassified exemplars are assigned lower weight and then given to the next randomly
generated decision stump(i,j). This process is repeated again for all the classes to get hypothesis weight for each
classifier.

Classification is done based on the majority vote by all the classifiers.

Best Result:

CONFUSION MATRIX
-----------------------------------------------------------------
		0		90		180		270
0		142		40		26		31
90		26		153		11		34
180		33		33		140		30
270		29		32		14		169
------------------------------------------------------------------
Accuracy = 64.0509013786%
Number of decision stumps: 70

Note: Since the i,j values are generated randomly, the accuracy of the classifier varies each time when the program is
is run
"""

from math import log
from math import exp
import random

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

# check the classifications considering the given attribute
def split_based_on_attribute(feature_vector, col1, col2, given_class):

    set1 = []
    set2 = []
    miss_classifications = 0
    classification_wt = 0
    correctly_classy_filenames = []
    wt_index = len(feature_vector[feature_vector.keys()[0]]) - 1

    for filename,features in feature_vector.iteritems():
        if int(features[col1]) > int(features[col2]):
            set1 += [filename]
            if int(features[0]) == int(given_class):
                classification_wt += feature_vector[filename][wt_index]
                correctly_classy_filenames += [filename]
            else:
                miss_classifications += 1
        else:
            set2 += [filename]
            if int(features[0]) != int(given_class):
                classification_wt += feature_vector[filename][wt_index]
                correctly_classy_filenames += [filename]
            else:
                miss_classifications += 1


    return (set1, set2, miss_classifications,classification_wt,correctly_classy_filenames)

# Train using AdaBoost
def train_boost(feature_vector, no_of_stumps):

    class_dict = {0:'0', 1:'90', 2:'180', 3:'270'}
    dict_stumps = {}

    original_length = len(feature_vector)

    for classes in class_dict:
        dict_stumps[classes] = {}
        current_weight = float(1) / float(len(feature_vector))

        # initialize weights
        for key in feature_vector:
            if len(feature_vector[key]) == original_length:
                feature_vector[key] += [current_weight]
            else:
                feature_vector[key][len(feature_vector[key]) - 1] = current_weight

        stumps = []
        for n in range(1, no_of_stumps + 1):
            var1 = random.randint(n + 1, len(feature_vector[feature_vector.keys()[0]]) - 2)
            var2 = random.randint(n + 1, len(feature_vector[feature_vector.keys()[0]]) - 2)
            stumps += [[var1, var2]]

        for m in range(0, no_of_stumps):
            classification_weight = -1
            i = stumps[m][0]
            j = stumps[m][1]
            set1, set2, miss_classify, new_class_wt, correct_files = split_based_on_attribute(feature_vector, i, j, class_dict[classes])

            if new_class_wt > classification_weight:
                classification_weight = new_class_wt
                error = float(miss_classify) / float(len(feature_vector))
                if error == 1:
                    error = 0.99999999
                elif error == 0:
                    error = 0.00000001
                hypothesis_weight = 0.5 * float(log((1 - error) / error))
                att1, att2 = i, j

            dict_stumps[classes][m] = (att1, att2, hypothesis_weight)

            # reassign weights and normalize
            wt_index = len(feature_vector[feature_vector.keys()[0]]) - 1
            total = 0
            for key in correct_files:
                feature_vector[key][wt_index] *= float((error/(1 - error)))

            for key in feature_vector:
                total += feature_vector[key][wt_index]

            for key in feature_vector:
                feature_vector[key][len(feature_vector[key]) - 1] = float(feature_vector[key][len(feature_vector[key]) - 1])\
                                                                                         / total

    return dict_stumps

# Classify after boosting using ensemble of stumps
def classifier(test_vector, classifiers, no_of_stumps):
    class_dict = {0: '0', 1: '90', 2: '180', 3: '270'}
    correctly_classified = 0
    confusion_dict = {'0':[0,0,0,0], '90':[0,0,0,0], '180':[0,0,0,0], '270':[0,0,0,0]}
    adaboost_output_file = {}
    for filename in test_vector:
        actual_orientation = test_vector[filename][0]
        ensemble_votes = []
        for key in classifiers:
            final_classified = 0
            for j in range(0, no_of_stumps):
                att1, att2, hyp_wt = classifiers[int(key)][j]
                if int(test_vector[filename][int(att1)]) > int(test_vector[filename][int(att2)]):
                    final_classified += hyp_wt
                else:
                    final_classified -= hyp_wt
            ensemble_votes += [final_classified]
        orientation = class_dict[ensemble_votes.index(max(ensemble_votes))]

        adaboost_output_file[filename] = orientation

        if orientation == actual_orientation:
            correctly_classified += 1
            if orientation == '0':
                confusion_dict[orientation][0] += 1
            elif orientation == '90':
                confusion_dict[orientation][1] += 1
            elif orientation == '180':
                confusion_dict[orientation][2] += 1
            else:
                confusion_dict[orientation][3] += 1

        if orientation == '0' and actual_orientation =='90':
            confusion_dict[actual_orientation][0] += 1
        elif orientation == '0' and actual_orientation =='180':
            confusion_dict[actual_orientation][0] += 1
        elif orientation == '0' and actual_orientation =='270':
            confusion_dict[actual_orientation][0] += 1
        elif orientation == '90' and actual_orientation =='0':
            confusion_dict[actual_orientation][1] += 1
        elif orientation == '90' and actual_orientation =='180':
            confusion_dict[actual_orientation][1] += 1
        elif orientation == '90' and actual_orientation =='270':
            confusion_dict[actual_orientation][1] += 1
        elif orientation == '180' and actual_orientation =='0':
            confusion_dict[actual_orientation][2] += 1
        elif orientation == '180' and actual_orientation =='90':
            confusion_dict[actual_orientation][2] += 1
        elif orientation == '180' and actual_orientation =='270':
            confusion_dict[actual_orientation][2] += 1
        elif orientation == '270' and actual_orientation =='0':
            confusion_dict[actual_orientation][3] += 1
        elif orientation == '270' and actual_orientation =='90':
            confusion_dict[actual_orientation][3] += 1
        elif orientation == '270' and actual_orientation =='180':
            confusion_dict[actual_orientation][3] += 1

    accuracy = float(correctly_classified) / float(len(test_vector))
    return accuracy * 100, confusion_dict, adaboost_output_file


# Read data from the file (specifically used for training data set)
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

# Read data from test file
def read_test_features(test_file):
    file = open(test_file, 'r')
    feature_vector = {}
    for line in file:
        list_features = []
        list_features += [feature for feature in line.split()]
        feature_vector[list_features[0]] = []
        feature_vector[list_features[0]] += list_features[1:]

    return feature_vector

# Write output to file
def output_file(adaboost_out):
    file = open("adaboost_output.txt", "w")
    for key in adaboost_out:
        file.write(str(key) + " " + str(adaboost_out[key]) + '\n')

# Read args
def read_data(train_file, test_file, num_stumps):
    print "running adaboost"
    train_file = train_file
    test_file = test_file
    no_of_stumps = num_stumps

    feature_dict = read_features(train_file)
    classifiers = train_boost(feature_dict, no_of_stumps)
    test_dict = read_test_features(test_file)
    accuracy, confusion_dict, adaboost_out = classifier(test_dict, classifiers, no_of_stumps)
    confusion_mat(confusion_dict, accuracy)
    output_file(adaboost_out)
