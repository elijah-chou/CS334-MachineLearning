import numpy as np
import pandas as pd
from perceptron import Perceptron, calc_mistakes
from perceptron import file_to_numpy
from sklearn.model_selection import KFold

def printMostPosAndNegW(model, vocabList):
    targetWeights = model.w[:-1]
    sortedInd = np.argsort(targetWeights)
    positiveInd = sortedInd[-15:].tolist()
    print("15 Words with Most Positive Weights:")
    for i in range(len(positiveInd)):
        print(vocabList[positiveInd[i]])

    negativeInd = sortedInd[:15].tolist()
    print("15 Words with Most Negative Weights:")
    for i in range(len(negativeInd)):
        print(vocabList[negativeInd[i]])



def main():
    """
    Main file to run from the command line.
    """
    

    #Hyperparameter tuning and report mistakes for binary dataset
    vocabList = list(pd.read_csv("binary_train_x.csv"))
    vocabList.pop(0)
    binary_xTrain = file_to_numpy("binary_train_x.csv")
    binary_yTrain = file_to_numpy("binary_train_y.csv")
    binary_xTest = file_to_numpy("binary_test_x.csv")
    binary_yTest = file_to_numpy("binary_test_y.csv")

    kf = KFold()
    optEpoch = 0
    maxMistakes = binary_xTrain.shape[0]
    # 1-100 smaller numbers, and test 100-1000 with bigger increments
    epochTest = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for maxEpoch in epochTest:
        model = Perceptron(maxEpoch)
        totalMistakes = 0
        for train_index, test_index in kf.split(binary_xTrain):
            X_train, X_test = binary_xTrain[train_index], binary_xTrain[test_index]
            y_train, y_test = binary_yTrain[train_index], binary_yTrain[test_index]
            model.train(X_train, y_train)
            yHat = model.predict(X_test)
            totalMistakes += calc_mistakes(yHat, y_test)
        totalMistakes /= 5.0
        print(totalMistakes)
        if maxMistakes > totalMistakes:
            maxMistakes = totalMistakes
            optEpoch = maxEpoch

    print("Optimal Epoch for Binary Dataset:")
    print(optEpoch)
    model = Perceptron(optEpoch)
    model.train(binary_xTrain, binary_yTrain)
    yHat = model.predict(binary_xTrain)
    print("Number of mistakes on the binary training set:")
    print(calc_mistakes(yHat, binary_yTrain))
    yHat = model.predict(binary_xTest)
    print("Number of mistakes on the binary test set:")
    print(calc_mistakes(yHat, binary_yTest))
    printMostPosAndNegW(model, vocabList)



    #Hyperparameter tuning and report mistakes for count dataset
    vocabList = list(pd.read_csv("count_train_x.csv"))
    vocabList.pop(0)
    count_xTrain = file_to_numpy("count_train_x.csv")
    count_yTrain = file_to_numpy("count_train_y.csv")
    count_xTest = file_to_numpy("count_test_x.csv")
    count_yTest = file_to_numpy("count_test_y.csv")

    optEpoch = 0
    maxMistakes = count_xTrain.shape[0]
    for maxEpoch in epochTest:
        model = Perceptron(maxEpoch+1)
        totalMistakes = 0
        for train_index, test_index in kf.split(count_xTrain):
            X_train, X_test = count_xTrain[train_index], count_xTrain[test_index]
            y_train, y_test = count_yTrain[train_index], count_yTrain[test_index]
            model.train(X_train, y_train)
            yHat = model.predict(X_test)
            totalMistakes += calc_mistakes(yHat, y_test)
        totalMistakes /= 5.0
        if maxMistakes > totalMistakes:
            maxMistakes = totalMistakes
            optEpoch = maxEpoch

    print("Optimal epoch for count dataset:")
    print(optEpoch)
    model = Perceptron(optEpoch)
    model.train(count_xTrain, count_yTrain)
    yHat = model.predict(count_xTrain)
    print("Number of mistakes on the count training set:")
    print(calc_mistakes(yHat, count_yTrain))
    yHat = model.predict(count_xTest)
    print("Number of mistakes on the count test set:")
    print(calc_mistakes(yHat, count_yTest))
    printMostPosAndNegW(model, vocabList)
            
        

if __name__ == "__main__":
    main()