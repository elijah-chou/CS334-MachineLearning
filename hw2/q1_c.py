import dt
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line


    parser = argparse.ArgumentParser()
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
    vary_maxDepth_acc = list()
    for i in range(40):
        dt1 = dt.DecisionTree('gini', i + 1, 10)
        trainAcc1, testAcc1 = dt.dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
        vary_maxDepth_acc.append([i + 1, trainAcc1, testAcc1])

    # compile accuracies and plot them.
    df = pd.DataFrame(vary_maxDepth_acc, columns=["maxDepth", 'Training Accuracy', 'Test Accuracy'])
    axes = plt.gca()
    df.plot(kind='line', x="maxDepth", y='Training Accuracy', ax=axes)
    df.plot(kind='line', x='maxDepth', y='Test Accuracy', ax=axes).get_figure().savefig('maxDepth_test_train_acc.png')

    vary_minSample_acc = list()
    for i in range(30):
        dt1 = dt.DecisionTree('gini', 15, i + 1)
        trainAcc1, testAcc1 = dt.dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
        vary_minSample_acc.append([i+1, trainAcc1, testAcc1])
    
    df1 = pd.DataFrame(vary_minSample_acc, columns=["minSample", 'Training Accuracy', 'Test Accuracy'])
    axes1 = plt.gca()
    axes.cla()
    df1.plot(kind='line', x="minSample", y='Training Accuracy', ax=axes1)
    df1.plot(kind='line', x='minSample', y='Test Accuracy', ax=axes1).get_figure().savefig('minSample_test_train_acc.png')

if __name__ == "__main__":
    main()