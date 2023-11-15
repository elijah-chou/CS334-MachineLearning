import knn
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import q4

def main():

    list_of_k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25, 27, 29, 31]

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
    accuracies = list()
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    for i in list_of_k:
        # no preprocessing
        acc1 = q4.knn_train_test(i, xTrain, yTrain, xTest, yTest)
        # preprocess the data using standardization scaling
        xTrainStd, xTestStd = q4.standard_scale(xTrain, xTest)
        acc2 = q4.knn_train_test(i, xTrainStd, yTrain, xTestStd, yTest)
        # preprocess the data using min max scaling
        xTrainMM, xTestMM = q4.minmax_range(xTrain, xTest)
        acc3 = q4.knn_train_test(i, xTrainMM, yTrain, xTestMM, yTest)
        # add irrelevant features
        xTrainIrr, xTestIrr = q4.add_irr_feature(xTrain, xTest)
        acc4 = q4.knn_train_test(i, xTrainIrr, yTrain, xTestIrr, yTest)
        accuracies.append((i, acc1, acc2, acc3, acc4))

    df = pd.DataFrame(accuracies, columns=["k", 'No Preprocessing', 'Standardized Scaling', 'MinMax Scaling', 'Add Irrelevant Features'])
    axes = plt.gca()
    df.plot(kind='line', x="k", y='No Preprocessing', ax=axes)
    df.plot(kind='line', x="k", y='Standardized Scaling', ax=axes)
    df.plot(kind='line', x="k", y='MinMax Scaling', ax=axes)
    df.plot(kind='line', x='k', y='Add Irrelevant Features', ax=axes).get_figure().savefig('preprocessing_acc.png')

if __name__ == "__main__":
    main()

