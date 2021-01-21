#!/usr/bin/env python

"""Foobar.py: ID3 implementation using only the basic Python libraries."""

__author__ = "Jarosław Zabuski"

import math
import random
from collections import Counter


class Row:
    def __init__(self, category, dictionary):
        self.category = category
        self.dictionary = dictionary


class Node:
    def __init__(self, name):
        self.name = name
        self.childDictionary = {}


setOfCategories = []


def main(filename, percentLearning):
    # input handling. attributes are self explanatory.
    input_file = open(filename, "r")
    headers = input_file.readline().strip().split(",")
    attributes = headers[1:]

    examples = []
    for line in input_file:
        line = line.strip().split(",")
        examples.append(line)

    # numberOfLearningRows - defines a size of the training set part, coming from the data set.
    numberOfLearningRows = int(len(examples) * (percentLearning / 100))
    random.shuffle(examples)

    global setOfCategories
    setOfCategories = set([i[0] for i in examples])

    listOfLearningDataRows = []
    listOfTrainingDataRows = []

    # Both for learning and testing set data, a list is created. This list for both of those contains each specific
    # data row with its corresponding headers and values, but in a way that helps in later usages. In
    # listOfLearningDataRows and listOfTestingDataRows, each row headers, except the target one, are enclosed with
    # their values for current row in a dictionary.

    count = 0
    for example in examples:
        category = example[0]
        dictionary = {}
        for num in range(1, len(headers)):
            dictionary[headers[num]] = example[num]
        if count < numberOfLearningRows:
            listOfLearningDataRows.append(Row(category, dictionary))
        else:
            listOfTrainingDataRows.append(Row(category, dictionary))
        count += 1

    # a decision tree is built at this moment, using randomly selected portion of data set.
    root = id3(listOfLearningDataRows, attributes)

    # prints newly created decision tree, meant for debugging.
    print_tree(root)

    # prints number of data rows used in each section of program. Printed to check whether or not percentLearning
    # functionality is working.
    print("Total number of data set rows to work with: " + str(len(examples)))
    print("Number of data set rows used for learning: " + str(len(listOfLearningDataRows)))
    print("Number of data set rows used for testing: " + str(len(listOfTrainingDataRows)))

    # tests how close the decision tree was to discovering the optimal routes.
    if len(listOfTrainingDataRows) == 0:
        test(listOfLearningDataRows, root)
    else:
        test(listOfTrainingDataRows, root)


# Calculates entropy of currently checked data column. Checks how many values can this column have and how many rows
# are classified by those values, to calculate final entropy.
def entropy(listOfDataRows):
    tempEntropy = 0
    for cat in setOfCategories:

        counter = 0
        for i in listOfDataRows:
            if i.category == cat:
                counter += 1

        count = counter / len(listOfDataRows)

        if count != 0:
            tempEntropy -= (count * math.log(count, 2))

    return tempEntropy


# Calculates the information gain for selected attribute in a data set.
def gain(S, A):
    tempGain = entropy(S)
    setOfValues = set([i.dictionary.get(A) for i in S])

    for value in setOfValues:

        tempList = []
        for i in S:
            if i.dictionary.get(A) == value:
                tempList.append(i)

        tempGain -= (len(tempList) / len(S) * entropy(tempList))

    return tempGain


# main ID3 implementation: base checks, finding a category with most information gain associated with it
def id3(listOfDataRows, listOfHeaders):
    # finds how many categories there are in the data set rows, using the imported Counter class. If there is only
    # one, that means that the data set is self-explanatory.
    exCats = []
    for i in listOfDataRows:
        exCats.append(i.category)
    catFreqs = Counter(exCats)

    if len(catFreqs) == 1:
        return Node(catFreqs.most_common(1)[0][0])
    if len(listOfHeaders) == 0:
        return Node(catFreqs.most_common(1)[0][0])

    greatestGain = 0
    attributeWithGreatestGain = None

    # finds the first most eligible attribute to classify data rows by.
    for attributes in listOfHeaders:
        g = gain(listOfDataRows, attributes)
        if g >= greatestGain:
            greatestGain = g
            attributeWithGreatestGain = attributes

    newNode = Node(attributeWithGreatestGain)
    setOfValues = set([i.dictionary.get(attributeWithGreatestGain) for i in listOfDataRows])
    listOfHeaders.remove(attributeWithGreatestGain)

    # Classifies the data to new nodes, corresponding to the values of currently the most eligible attribute. If
    # needed (data set is still not pure or pure enough) will call recursively to find another attribute to
    # sort data rows by. If all remaining data rows are fully described by the new attribute and its value,
    # then saves it into a leaf node.
    for value in setOfValues:
        newlistOfDataRows = []

        for i in listOfDataRows:
            if i.dictionary.get(attributeWithGreatestGain) == value:
                newlistOfDataRows.append(i)

        if len(newlistOfDataRows) == 0:
            newNode.name = catFreqs.most_common(1)[0][0]

            return newNode
        newNode.childDictionary[value] = id3(newlistOfDataRows, listOfHeaders)

    return newNode


# Prints in a rather cute manner the constructed decision tree.
def print_tree(node, level=0):
    if len(node.childDictionary) == 0:
        print("  |  " * level + node.name)

    else:
        print("  |  " * level + node.name + "?")

    for childKey in node.childDictionary:
        print(("  |  " * (level + 1)) + childKey + ":")
        print_tree(node.childDictionary[childKey], level + 2)


# Tests the constructed decision tree. Starts at root and goes through the possible routes until the visited node is
# a category (leaf node) and checks, whether or not the category and its value corresponds with the classification
# the decision tree gave. Computes the accuracy by counting positively categorised rows.
def test(listOfDataRows, root):
    correct = 0
    for i in listOfDataRows:
        if test_node(i, root) is True:
            correct += 1

    percentCompliance = (correct / len(listOfDataRows)) * 100
    print("% of compliance with original data = {:.2f}".format(percentCompliance))


# Called for every test data set row, checks whether or not the decision tree properly categorises elements or not.
def test_node(ex, node):
    if ex.dictionary[node.name] not in node.childDictionary:
        # print("ERROR: value not in tree: " + ex.dictionary[node.name])
        return False

    node = node.childDictionary[ex.dictionary[node.name]]

    if len(node.childDictionary) == 0:
        if node.name == ex.category:
            return True

        else:
            return False

    return test_node(ex, node)


# TODO: input data set and percentLearning from console
if __name__ == "__main__":
    main("mushroom.txt", 5)
