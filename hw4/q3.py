import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from perceptron import file_to_numpy
from perceptron import calc_mistakes

def main():
    """
    Main file to run from the command line.
    """
    binary_xTrain = file_to_numpy("binary_train_x.csv")
    binary_yTrain = file_to_numpy("binary_train_y.csv")
    binary_xTest = file_to_numpy("binary_test_x.csv")
    binary_yTest = file_to_numpy("binary_test_y.csv")

    binary_xTrain = np.delete(binary_xTrain, 0, 1)
    binary_xTest = np.delete(binary_xTest, 0, 1)
    binary_yTrain = np.delete(binary_yTrain, 0, 1)
    binary_yTrain = np.ravel(binary_yTrain)

    clf = MultinomialNB().fit(binary_xTrain, binary_yTrain)
    print("Number of mistakes made by Multinomial Naive Bayes on Binary Test Set:")
    yHat = clf.predict(binary_xTest)
    print(calc_mistakes(yHat, binary_yTest))

    clf = LogisticRegression().fit(binary_xTrain, binary_yTrain)
    print("Number of mistakes made by Logistic Regression on Binary Test Set:")
    yHat = clf.predict(binary_xTest)
    print(calc_mistakes(yHat, binary_yTest))


    count_xTrain = file_to_numpy("count_train_x.csv")
    count_yTrain = file_to_numpy("count_train_y.csv")
    count_xTest = file_to_numpy("count_test_x.csv")
    count_yTest = file_to_numpy("count_test_y.csv")

    count_xTrain = np.delete(count_xTrain, 0, 1)
    count_xTest = np.delete(count_xTest, 0, 1)
    count_yTrain = np.delete(count_yTrain, 0, 1)
    count_yTrain = np.ravel(count_yTrain)

    clf = MultinomialNB().fit(count_xTrain, count_yTrain)
    print("Number of mistakes made by Multinomial Naive Bayes on Count Test Set:")
    yHat = clf.predict(count_xTest)
    print(calc_mistakes(yHat, count_yTest))

    clf = LogisticRegression().fit(count_xTrain, count_yTrain)
    print("Number of mistakes made by Logistic Regression on Count Test Set:")
    yHat = clf.predict(count_xTest)
    print(calc_mistakes(yHat, count_yTest))


if __name__ == "__main__":
    main()