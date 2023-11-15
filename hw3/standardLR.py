import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        start = time.time()
        trainStats = {}
        iterNum = 0
        #Training stage
        ones = np.ones((xTrain.shape[0], 1))
        xTrainOnes = np.concatenate((ones, xTrain), axis=1)
        xTrainTrans = np.transpose(xTrainOnes)
        self.beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(xTrainTrans, xTrainOnes)), xTrainTrans), yTrain)

        #Predict stage
        trainMSE = self.mse(xTrainOnes, yTrain)
        ones = np.ones((xTest.shape[0], 1))
        xTestOnes = np.concatenate((ones, xTest), axis=1)
        testMSE = self.mse(xTestOnes, yTest)
        end = time.time()
        totalTime = end - start
        trainStats[iterNum] = {'time':totalTime, 'train-mse': trainMSE, 'test-mse': testMSE}
        iterNum = iterNum + 1
        return trainStats


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

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
