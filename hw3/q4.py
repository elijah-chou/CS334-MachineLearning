import matplotlib.pyplot as plt
import sgdLR
import lr
from standardLR import StandardLR

def main():
    xTrain = lr.file_to_numpy('new_xTrain.csv')
    yTrain = lr.file_to_numpy('eng_yTrain.csv')
    xTest = lr.file_to_numpy('new_xTest.csv')
    yTest = lr.file_to_numpy('eng_yTest.csv')
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    batchSize = [1, 10, 15, 30, 559, 1118, 16770]
    for size in batchSize:
        model = sgdLR.SgdLR(0.001, size, 5)
        results = model.train_predict(xTrain, yTrain, xTest, yTest)
        for iterNum in range(len(results)):
            trainingResults = []
            trainingTime = []
            testResults = []
            trainingResults.append(results[iterNum]['train-mse'])
            testResults.append(results[iterNum]['test-mse'])
            trainingTime.append(results[iterNum]['time'])
        ax1.scatter(trainingTime, trainingResults, label=size)
        ax2.scatter(trainingTime, testResults, label=size)
    model = StandardLR()
    closedForm = model.train_predict(xTrain, yTrain, xTest, yTest)
    ax1.scatter(closedForm[0]['time'], closedForm[0]['train-mse'], label='Closed Form Sol.')
    ax2.scatter(closedForm[0]['time'], closedForm[0]['test-mse'], label='Closed Form Sol.')
    ax1.set_title('Train-MSE vs. Total Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('MSE')
    ax2.set_title('Test-MSE vs. Total Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('MSE')

    plt.legend()
    plt.savefig('batch_size_effect.png')
        


if __name__ == "__main__":
    main()