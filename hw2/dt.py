import argparse
import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class Node():
    left = None
    right = None
    currentDepth = 0
    maxDepth = 0
    colNum = 0
    splitVal = 0
    criterion = None
    minLeafSample = 0
    dataPoints = list()
    finalLabel = -1
    leftSplitList = list()
    rightSplitList = list()

    def __init__(self, currentDepth, maxDepth, criterion, minLeafSample):
        self.currentDepth = currentDepth
        self.maxDepth = maxDepth
        self.criterion = criterion
        self.minLeafSample = minLeafSample

def giniIndex(left, right):
    allGroups = [left, right]
    totalN = len(left) + len(right)
    giniScore = 0.0
    for group in allGroups:
        size = len(group)
        if size == 0:
            continue
        groupScore = 0.0
        for label in range(2):
            p = [row[-1] for row in group].count(label) / size
            groupScore += p * p
        giniScore += (1.0 - groupScore) * (size/totalN)
    return giniScore

def entropy(node, left, right):
    parentEntropy = 0.0
    balanceEntropy = 0.0
    allGroups = [left, right]
    for label in range(2):
        ratio = [row[-1] for row in node.dataPoints].count(label) / len(node.dataPoints) 
        parentEntropy -= ratio * math.log(ratio, 2)

    for group in allGroups:
        if len(group) == 0:
            continue
        branchEntropy = 0.0
        for label in range(2):
            ratio = [row[-1] for row in group].count(label) / len(group)
            if ratio > 0:
                branchEntropy -= ratio * math.log(ratio, 2)
        balanceEntropy += (len(group) / len(node.dataPoints)) * branchEntropy
    
    infoGain = parentEntropy - balanceEntropy
    return infoGain

def getSplit(parentNode):
    colNum, value, leftList, rightList = 0, 0, list(), list()
    score = None
    if(parentNode.criterion == "gini"):
        score = 1000
    elif(parentNode.criterion == "entropy"):
        score = 0
    for attr in range(len(parentNode.dataPoints[0])-1):
        for splitPoint in parentNode.dataPoints:
            #The following is used to split at every possible point and calculate Gini Index
            left, right = list(), list()
            for row in parentNode.dataPoints:
                if row[attr] < splitPoint[attr]:
                    left.append(row)
                else:
                    right.append(row)
            #If criterion is "Gini Index", then we want to minimize gini index to get best split
            if(parentNode.criterion == "gini"):
                gini = giniIndex(left, right)
                if gini < score:
                    colNum = attr
                    value = splitPoint[attr]
                    score = gini
                    leftList = left
                    rightList = right
            #If criterion is "Entropy", then we want to find split that helps us get the largest information gain
            if(parentNode.criterion == "entropy"):
                infoGain = entropy(parentNode, left, right)
                if infoGain > score:
                    colNum = attr
                    value = splitPoint[attr]
                    score = infoGain
                    leftList = left
                    rightList = right
    return colNum, value, leftList, rightList

def determineFinalLabel(group):
    labels = [row[-1] for row in group]
    return max(set(labels), key=labels.count)

def split(node, maxDepth, criterion, minSize, currentDepth):
    node.left = Node(currentDepth+1, maxDepth, criterion, minSize)
    node.right = Node(currentDepth+1, maxDepth, criterion, minSize)
    node.left.dataPoints = node.leftSplitList
    node.right.dataPoints = node.rightSplitList
    if not node.left.dataPoints or not node.right.dataPoints:
        node.left.finalLabel = node.right.finalLabel = determineFinalLabel(node.left.dataPoints+ node.right.dataPoints)
        return
    if currentDepth >= maxDepth:
        node.left.finalLabel = determineFinalLabel(node.left.dataPoints)
        node.right.finalLabel = determineFinalLabel(node.right.dataPoints)
        return
    if len(node.left.dataPoints) <= minSize:
        node.left.finalLabel = determineFinalLabel(node.left.dataPoints)
    else:
        node.left.colNum, node.left.splitVal, node.left.leftSplitList, node.left.rightSplitList = getSplit(node.left)
        split(node.left, maxDepth, criterion, minSize, currentDepth+1)
    if len(node.right.dataPoints) <= minSize:
        node.right.finalLabel = determineFinalLabel(node.right.dataPoints)
    else:
        node.right.colNum, node.right.splitVal, node.right.leftSplitList, node.right.rightSplitList = getSplit(node.right)
        split(node.right, maxDepth, criterion, minSize, currentDepth+1)
    
def predictLabel(node, testPoint):
    if node.left is None and node.right is None:
        return node.finalLabel
    if testPoint[node.colNum] < node.splitVal:
        return predictLabel(node.left, testPoint)
    else:
        return predictLabel(node.right, testPoint)


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    root = None

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        
    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        self.root = Node(0, self.maxDepth, self.criterion, self.minLeafSample)
        xFeat["labels"] = y
        self.root.dataPoints = xFeat.values.tolist()
        self.root.colNum, self.root.splitVal, self.root.leftSplitList, self.root.rightSplitList = getSplit(self.root)
        split(self.root, self.root.maxDepth, self.root.criterion, self.root.minLeafSample, 0)
        return self
    



    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # TODO
        testPoints = xFeat.values.tolist()
        for point in testPoints:
            predictedLabel = predictLabel(self.root, point)
            yHat.append(predictedLabel)
        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
