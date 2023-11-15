from rf import RandomForest
from rf import file_to_numpy
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

def main():
    
    xTest = file_to_numpy("q4xTest.csv")
    xTrain = file_to_numpy("q4xTrain.csv")
    yTest = file_to_numpy("q4yTest.csv")
    yTrain = file_to_numpy("q4yTrain.csv")

    nestTest = [1, 5, 9, 15, 20, 25]
    maxFeatTest = [3, 5, 7, 9, 11]
    criterionTest = ["gini", "entropy"]
    maxDepthTest = [6, 8, 10, 12, 14]
    minLeafSampleTest = [2, 4, 6, 8, 10]
    minError = 1
    optNest = 0
    optFeat = 0
    optCrit = None
    optDepth = 0
    optMinLeaf = 0

    #Testing optimal parameter for "nest"
    nestError = []
    for nest in nestTest:
        model = RandomForest(nest, 5, "gini", 5, 5)
        error = model.train(xTrain, yTrain)
        nestError.append(error[nest-1])
    pyplot.plot(nestTest, nestError, label = "Nest")
    pyplot.xlabel("Nest Value")
    pyplot.ylabel("OOB Error")
    pyplot.legend()
    pyplot.show()

    #Testing optimal parameter for "maxFeat"
    featError = []
    for maxFeat in maxFeatTest:
        model = RandomForest(10, maxFeat, "gini", 5, 5)
        error = model.train(xTrain, yTrain)
        featError.append(error[9])
    pyplot.plot(maxFeatTest, featError, label = "maxFeat")
    pyplot.xlabel("maxFeat Value")
    pyplot.ylabel("OOB Error")
    pyplot.legend()
    pyplot.show()

    #Testing optimal parameter for "criterion"
    for criterion in criterionTest:
        model = RandomForest(10, 5, criterion, 5, 5)
        error = model.train(xTrain, yTrain)
        print("%s OOB error: %d", criterion, error[9])
    
    #Testing optimal parameter for "maxDepth"
    depthError = []
    for maxDepth in maxDepthTest:
        model = RandomForest(10, 5, "gini", maxDepth, 5)
        error = model.train(xTrain, yTrain)
        depthError.append(error[9])
    pyplot.plot(maxDepthTest, depthError, label = "maxDepth")
    pyplot.xlabel("maxDepth Value")
    pyplot.ylabel("OOB Error")
    pyplot.legend()
    pyplot.show()

    #Testing optimal parameter for "minSampleLeaf"
    minLeafError = []
    for minLeafSample in minLeafSampleTest:
        model = RandomForest(10, 5, "gini", 5, minLeafSample)
        error = model.train(xTrain, yTrain)
        minLeafError.append(error[9])
    pyplot.plot(minLeafSampleTest, minLeafError, label = "minLeafSample")
    pyplot.xlabel("minLeafSample Value")
    pyplot.ylabel("OOB Error")
    pyplot.legend()
    pyplot.show()

    # #final all test
    # for nest in nestTest:
    #     for maxFeat in maxFeatTest:
    #         for criterion in criterionTest:
    #             for maxDepth in maxDepthTest:
    #                 for minLeafSample in minLeafSampleTest:
    #                     model = RandomForest(nest, maxFeat, criterion, maxDepth, minLeafSample)
    #                     error = model.train(xTrain, yTrain)
    #                     if error[nest-1] < minError:
    #                         minError = error[nest-1]
    #                         optNest = nest
    #                         optFeat = maxFeat
    #                         optCrit = criterion
    #                         optDepth = maxDepth
    #                         optMinLeaf = minLeafSample
    # print("OOB Error of optimal model is ", minError)
    # print("Optimal Nest is ", optNest)
    # print("Optimal maxFeat is ", optFeat)
    # print("Optimal criterion is ", optCrit)
    # print("Optimal depth is ", optDepth)
    # print("Optimal minLeafSample is ", optMinLeaf)
    # model = RandomForest(optNest, optFeat, optCrit, optDepth, optMinLeaf)
    # yHat = model.predict(xTest)
    # print("Accuracy of optimal model is:")
    # print(accuracy_score(yTest, yHat))

    #Optimal parameter run
    model = RandomForest(10, 5, "gini", 10, 4)
    error = model.train(xTrain, yTrain)
    print("OOB Error of Optimal Model is ")
    print(error[9])
    yHat = model.predict(xTest)
    print("Accuracy of optimal model is:")
    print(accuracy_score(yTest, yHat))


    


if __name__ == "__main__":
    main()