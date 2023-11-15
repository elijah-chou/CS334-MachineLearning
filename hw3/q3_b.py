import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sgdLR
import lr

def main():
    xTrain = lr.file_to_numpy('new_xTrain.csv')
    yTrain = lr.file_to_numpy('eng_yTrain.csv')
    xTest = lr.file_to_numpy('new_xTest.csv')
    yTest = lr.file_to_numpy('eng_yTest.csv')

    xTrain40, xTest40, yTrain40, yTest40 = train_test_split(xTrain, yTrain, train_size=0.4)
    learningRates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    for rate in learningRates:
        model = sgdLR.SgdLR(rate, 1, 10)
        results = model.train_predict(xTrain40, yTrain40, xTest, yTest)
        trainSize = xTrain40.shape[0]
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = []
        y.append(results[0]['train-mse'])
        for iterNum in range(len(x)-1):
            y.append(results[((iterNum+1)*trainSize)-1]['train-mse'])
        plt.plot(x, y, label = rate)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Train-MSE')
    plt.savefig('trainmse_vs_epoch.png')
        


if __name__ == "__main__":
    main()




