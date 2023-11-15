import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    forest = []
    oobList = {}
    featureList = {}

    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest #test a couple, but test more than just 5
        self.maxFeat = maxFeat #test with value of up to 10, do about 5 test values
        self.criterion = criterion
        self.maxDepth = maxDepth #test with value of up to 10, do about 5 test values
        self.minLeafSample = minLeafSample #test with value of up to 10, do about 5 test values
        self.forest = []
        self.oobList = {}
        self.featureList = {}

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        results = {}
        for b in range(self.nest):
            clf = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth, min_samples_leaf=self.minLeafSample,
            max_features=self.maxFeat)
            row_indices = np.arange(xFeat.shape[0])
            feature_indices = np.arange(xFeat.shape[1])
            num_features = np.random.choice(feature_indices, size=self.maxFeat, replace=False) #choose random subset of features
            ran_indices = np.random.choice(row_indices, size=xFeat.shape[0], replace=True) #choose random subset of points w/ replacement
            bootstrapSample = xFeat[ran_indices, :]
            bootstrapLabels = y[ran_indices]
            trainingSubset = bootstrapSample[:, num_features]
            clf.fit(trainingSubset, bootstrapLabels)
            self.forest.append(clf)
            self.oobList[b] = []
            self.featureList[b] = num_features
            num_features_list = num_features.tolist()
            for i in range(xFeat.shape[0]):
                if i not in num_features_list:
                    self.oobList[b].append(i)
            #calculating OOB error for all trees up to now
            totalError = 0
            total_instances = 0
            for tree in range(len(self.forest)):
                for index in range(xFeat.shape[0]):
                    if index not in self.oobList[tree]:
                        yHat = self.forest[tree].predict(xFeat[index, self.featureList[tree]].reshape(1, -1))
                        if yHat != y[index]:
                            totalError += 1
                        total_instances += 1
            results[b] = totalError/total_instances
        return results

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
            Predicted response per sample
        """
        #add up and then if >=0.5 then 1, if <0.5 then 0
        yHat = []
        for index in range(xFeat.shape[0]):
            total = 0.0
            for tree in range(len(self.forest)):
                total += self.forest[tree].predict(xFeat[index, self.featureList[tree]].reshape(1, -1))
            total /= len(self.forest)
            if total >= 0.5:
                yHat.append(1)
            else:
                yHat.append(0)
        return yHat


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = RandomForest(30, 5, "gini", 5, 5)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    print(yHat)


if __name__ == "__main__":
    main()