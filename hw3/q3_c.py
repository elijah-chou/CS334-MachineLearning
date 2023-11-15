import matplotlib.pyplot as plt
import sgdLR
import lr

def main():
    xTrain = lr.file_to_numpy('new_xTrain.csv')
    yTrain = lr.file_to_numpy('eng_yTrain.csv')
    xTest = lr.file_to_numpy('new_xTest.csv')
    yTest = lr.file_to_numpy('eng_yTest.csv')
    model = sgdLR.SgdLR(0.001, 1, 10)
    results = model.train_predict(xTrain, yTrain, xTest, yTest)
    trainSize = xTrain.shape[0]
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    yTrainMSE = []
    yTrainMSE.append(results[0]['train-mse'])
    yTestMSE = []
    yTestMSE.append(results[0]['test-mse'])
    for iterNum in range(len(x)-1):
        yTrainMSE.append(results[((iterNum+1)*trainSize)-1]['train-mse'])
        yTestMSE.append(results[((iterNum+1)*trainSize)-1]['test-mse'])
    plt.plot(x, yTrainMSE, label = 'Train-MSE')
    plt.plot(x, yTestMSE, label='Test-MSE')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Train-MSE')
    plt.savefig('train_test_mse_vs_epoch.png')
        


if __name__ == "__main__":
    main()