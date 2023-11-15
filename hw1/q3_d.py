import knn
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():

    list_of_k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 31, 41, 51]

    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")
    args = parser.parse_args()
    accuracies = list()
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    knn_model = knn.Knn(1)
    knn_model.train(xTrain, yTrain['label'])


    for i in list_of_k:
        knn_model.k = i
        yHatTrain = knn_model.predict(xTrain)
        train_acc = knn.accuracy(yHatTrain, yTrain['label'])
        yHatTest = knn_model.predict(xTest)
        test_acc = knn.accuracy(yHatTest, yTest['label'])
        accuracies.append((i, train_acc, test_acc))

    df = pd.DataFrame(accuracies, columns=["k", 'Training Accuracy', 'Test Accuracy'])
    axes = plt.gca()
    df.plot(kind='line', x="k", y='Training Accuracy', ax=axes)
    df.plot(kind='line', x='k', y='Test Accuracy', ax=axes).get_figure().savefig('test_train_acc.png')

if __name__ == "__main__":
    main()




