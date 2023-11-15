import argparse
import numpy as np
from numpy.core.fromnumeric import transpose
import time

from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        start = time.time()
        trainStats = {}
        self.beta = np.zeros((xTrain.shape[1]+1, 1)) #initialize beta values to 0
        ones = np.ones((xTrain.shape[0], 1))
        xTrain = np.concatenate((ones, xTrain), axis=1)
        ones = np.ones((xTest.shape[0], 1))
        xTest = np.concatenate((ones, xTest), axis=1)
        iterNum = 0
        #SGD Algorithm
        for epoch in range(self.mEpoch):
            combinedArray = np.concatenate((xTrain, yTrain), axis=1)
            rng = np.random.default_rng()
            rng.shuffle(combinedArray)
            batches = np.split(combinedArray, xTrain.shape[0]/self.bs) #split shuffled xTrain and yTrain into batches
            for batch in range(len(batches)):
                gradientTotal = np.zeros((batches[batch].shape[1]-1, 1))
                for row in range(batches[batch].shape[0]):
                    sample = batches[batch]
                    gradientTotal = gradientTotal + np.transpose(sample[row,:-1]) * (sample[row, sample.shape[1]-1] - np.matmul(sample[row,:-1], self.beta))
                gradientAverage = gradientTotal/self.bs
                self.beta = self.beta + self.lr * gradientAverage
                self.beta = self.beta[0, :]
                trainMSE = self.mse(xTrain, yTrain)
                testMSE = self.mse(xTest, yTest)
                end = time.time()
                totalTime = end - start
                trainStats[iterNum] = {'time': totalTime, 'train-mse':trainMSE, 'test-mse':testMSE}     
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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

