# Jarosław Zabuski, Jakub Strawa

import math
import random
from collections import Counter


# class Row: category - edible attribute values,
# dictionary - all attributes (except edible) with their values
# one row describes one mushroom
class Row:
    def __init__(self, category, dictionary):
        self.category = category
        self.dictionary = dictionary


# class Node: name - name of attribute,
# childrenDictionary - dictionary of attribute values and corresponding children nodes
# if node has no children, then node name is either EDIBLE or POISONOUS and childrenDictionary is empty
class Node:
    def __init__(self, name):
        self.name = name
        self.childrenDictionary = {}


# class ID3: attributes - list with all yet unused attributes except first one - edible,
# learningDataSet - list of rows with learning data,
# testingDataSet - list of rows used for testing, root - root node in decision tree,
# percentLearning - percentage of data used for learning
# categories - all possible categories decision tree can predict - in case of mushrooms - if edible
# max_tree_depth - used only for testing
class ID3:
    def __init__(self, percent=5):
        self.attributes = []
        self.learningDataSet = []
        self.testingDataSet = []
        self.root = None
        self.percentLearning = percent
        self.categories = {}
        self.max_tree_depth = 0

    # setup function for reading data and preparing variables
    def prepare_data(self, filename="mushroom.txt"):
        # input handling
        input_file = open(filename, "r")
        headers = input_file.readline().strip().split(",")
        self.attributes = headers[1:]

        # examples - list containing all mushroom data
        examples = []
        for line in input_file:
            line = line.strip().split(",")
            examples.append(line)

        input_file.close()
        # create set with all possible category values
        self.categories = set([i[0] for i in examples])

        # numberOfLearningRows - defines a size of the training set
        numberOfLearningRows = int(len(examples) * (self.percentLearning / 100))
        # shuffle all data rows
        random.shuffle(examples)

        count = 0
        for example in examples:
            category = example[0]
            dictionary = {}
            # creating dictionary of attributes and values for each row
            for num in range(1, len(headers)):
                dictionary[headers[num]] = example[num]
            # adding given number of rows to learning set and rest to training set
            if count < numberOfLearningRows:
                self.learningDataSet.append(Row(category, dictionary))
            else:
                self.testingDataSet.append(Row(category, dictionary))
            count += 1

    # main ID3 implementation: base checks, finding a category with most information gain associated with it
    def classic_ID3(self, dataRows, unusedAttributes):
        # finds how many categories there are in the data set rows
        # if there is only one, that means that the data set is self-explanatory
        exAttr = []
        for row in dataRows:
            exAttr.append(row.category)
        catFreqs = Counter(exAttr)

        if len(catFreqs) == 1:
            return Node(catFreqs.most_common(1)[0][0])
        if len(unusedAttributes) == 0:
            return Node(catFreqs.most_common(1)[0][0])

        greatestGain = 0
        attributeWithGreatestGain = None

        # finds the first most eligible attribute to classify data rows by.
        for attribute in unusedAttributes:
            g = self.gain(dataRows, attribute)
            if g >= greatestGain:
                greatestGain = g
                attributeWithGreatestGain = attribute

        newNode = Node(attributeWithGreatestGain)
        setOfValues = set([row.dictionary.get(attributeWithGreatestGain) for row in dataRows])
        unusedAttributes.remove(attributeWithGreatestGain)

        # classifies the data to new nodes, according to the values of currently the most fitted attribute
        # if data set is still not used, will call recursively to find another attribute to sort data by
        # if all left data rows are fully described by the new attribute and its value, then saves it into a leaf node
        for value in setOfValues:
            newlistOfDataRows = []

            for row in dataRows:
                if row.dictionary.get(attributeWithGreatestGain) == value:
                    newlistOfDataRows.append(row)

            if len(newlistOfDataRows) == 0:
                newNode.name = catFreqs.most_common(1)[0][0]

                return newNode
            newNode.childrenDictionary[value] = self.classic_ID3(newlistOfDataRows, unusedAttributes)

        return newNode

    # modified ID3 implementation, uses roulette system based on info gain to draw next node in decision tree
    def modified_ID3(self, dataRows, unusedAttributes):
        # finds how many categories there are in the data set rows
        # if there is only one, that means that the data set is self-explanatory
        exAttr = []
        for row in dataRows:
            exAttr.append(row.category)
        catFreqs = Counter(exAttr)

        if len(catFreqs) == 1:
            return Node(catFreqs.most_common(1)[0][0])
        if len(unusedAttributes) == 0:
            return Node(catFreqs.most_common(1)[0][0])

        # list contains all gains
        gainsList = []

        # finds the first most eligible attribute to classify data rows by.
        for attribute in unusedAttributes:
            g = self.gain(dataRows, attribute)
            gainsList.append(g)

        # if all left attributes have no info gain, return most frequent left attribute of parent
        if sum(gainsList) == 0:
            return Node(catFreqs.most_common(1)[0][0])

        # randomly choose which attribute to use
        pickedGain = random.uniform(0.0, sum(gainsList))
        pickedAttribute = ""
        for i in range(0, len(gainsList)):
            pickedGain -= gainsList[i]
            if pickedGain <= 0:
                pickedAttribute = unusedAttributes[i]
                break

        newNode = Node(pickedAttribute)
        setOfValues = set([row.dictionary.get(pickedAttribute) for row in dataRows])
        unusedAttributes.remove(pickedAttribute)

        # classifies the data to new nodes, according to the values of currently the most fitted attribute
        # if data set is still not used, will call recursively to find another attribute to sort data by
        # if all left data rows are fully described by the new attribute and its value, then saves it into a leaf node
        for value in setOfValues:
            newlistOfDataRows = []

            for row in dataRows:
                if row.dictionary.get(pickedAttribute) == value:
                    newlistOfDataRows.append(row)

            if len(newlistOfDataRows) == 0:
                newNode.name = catFreqs.most_common(1)[0][0]
                return newNode

            newNode.childrenDictionary[value] = self.modified_ID3(newlistOfDataRows, unusedAttributes)

        return newNode

    # prints decision tree
    def print_tree(self, node, level=0):
        if self.max_tree_depth < level:
            self.max_tree_depth = level
        if len(node.childrenDictionary) == 0:
            print("  |  " * level + node.name)

        else:
            print("  |  " * level + node.name + "?")

        for childKey in node.childrenDictionary:
            print(("  |  " * (level + 1)) + childKey + ":")
            self.print_tree(node.childrenDictionary[childKey], level + 2)

    # tests how accurate the decision tree is and returns percentage
    def test_tree(self):
        if len(self.testingDataSet) == 0:
            print("You should not train your model with all your data!")
            percent = self.test(self.learningDataSet, self.root)
            return percent
        else:
            percent = self.test(self.testingDataSet, self.root)
            return percent

    # testing decision tree. Starts at root and goes through all the possible routes until the visited node is
    # an attribute and checks, whether or not the category and its value is the same as the one in decision tree.
    def test(self, dataRows, root):
        correct = 0
        for i in dataRows:
            if self.test_node(i, root) is True:
                correct += 1

        percentCompliance = (correct / len(dataRows)) * 100
        return percentCompliance

    # called for every test data set row, checks whether or not the decision tree properly categorises elements or not
    def test_node(self, ex, node):
        if ex.dictionary[node.name] not in node.childrenDictionary:
            return False

        node = node.childrenDictionary[ex.dictionary[node.name]]

        # if node has no children, check if true or false
        if len(node.childrenDictionary) == 0:
            if node.name == ex.category:
                return True
            else:
                return False

        return self.test_node(ex, node)

    # calculates entropy of currently checked data
    def entropy(self, dataRows):
        tempEntropy = 0
        # count all rows with given category and calculate entropy
        for category in self.categories:
            counter = 0
            for row in dataRows:
                if row.category == category:
                    counter += 1

            count = counter / len(dataRows)
            if count != 0:
                tempEntropy -= (count * math.log(count, 2))

        return tempEntropy

    # calculates the information gain for selected attribute
    def gain(self, dataRows, attribute):
        tempGain = self.entropy(dataRows)
        setOfValues = set([i.dictionary.get(attribute) for i in dataRows])

        for value in setOfValues:
            tempList = []
            for row in dataRows:
                if row.dictionary.get(attribute) == value:
                    tempList.append(row)
            tempGain -= (len(tempList) / len(dataRows) * self.entropy(tempList))

        return tempGain
