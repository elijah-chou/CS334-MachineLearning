import argparse
import numpy as np
import pandas as pd
import time

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        self.w = np.zeros(xFeat.shape[1])
        prediction = 0
        for runNum in range(self.mEpoch):
            numMistakes = 0
            for row in range(xFeat.shape[0]):
                if (np.dot(self.w[:-1], xFeat[row,1:]) + self.w[-1]) >= 0:
                    prediction = 1
                else:
                    prediction = 0
                if y[row,1] == prediction:
                    continue
                else:
                    if prediction == 0:
                        self.w[:-1] = self.w[:-1] + xFeat[row, 1:]
                        self.w[-1] = self.w[-1] + 1
                    elif prediction == 1:
                        self.w[:-1] = self.w[:-1] - xFeat[row, 1:]
                        self.w[-1] = self.w[-1] - 1
                    numMistakes += 1
            stats[runNum] = numMistakes
            if numMistakes == 0:
                break
        return stats

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
        yHat = []
        for row in range(xFeat.shape[0]):
            if (np.dot(self.w[:-1], xFeat[row, 1:]) + self.w[-1]) >= 0:
                yHat.append(1)
            else:
                yHat.append(0)
        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    numMistakes = 0
    for point in range(len(yHat)):
        if yHat[point] != yTrue[point, 1]:
            numMistakes += 1
    return numMistakes


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
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy("binary_train_x.csv")
    yTrain = file_to_numpy("binary_train_y.csv")
    xTest = file_to_numpy("binary_test_x.csv")
    yTest = file_to_numpy("binary_test_y.csv")

    np.random.seed(args.seed)   
    model = Perceptron(1000)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))


if __name__ == "__main__":
    main()