import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

#Part A: k-fold validation to find optimal hyperparameters for k-nn and decision tree.
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
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    xTrain5, xTest5, yTrain5, yTest5 = train_test_split(xTrain, yTrain, train_size=0.95)
    xTrain10, xTest10, yTrain10, yTest10 = train_test_split(xTrain, yTrain, train_size= 0.9)
    xTrain20, xTest20, yTrain20, yTest20 = train_test_split(xTrain, yTrain, train_size=0.8)
    datasets = [[xTrain, yTrain], [xTrain5, yTrain5], [xTrain10, yTrain10], [xTrain20, yTrain20]]
    #Part A: k-fold validation to find optimal hyperparameters for k-nn and decision tree.
    testRange = np.arange(50) + 1
    testRange = testRange.tolist()
    param_grid = {'max_depth': testRange,
                'min_samples_leaf':testRange,
                'criterion': ['gini', 'entropy']}

    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit=True, cv=5)
    grid.fit(xTrain, yTrain)
    print(grid.best_params_)
    criterion = grid.best_params_['criterion']
    max_depth = grid.best_params_['max_depth']
    min_samples_leaf = grid.best_params_['min_samples_leaf']

    param_grid = {'n_neighbors': testRange}
    grid1 = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, cv=5)
    grid1.fit(xTrain, yTrain)
    print(grid1.best_params_)
    n_neighbors = grid1.best_params_['n_neighbors']

    #Part B: Use best params for k-nn algorithm to train 1) entire dataset,
    # 2) 5% removed, 3) 10% removed, and 4) 20% removed. Then evaluate AUC and
    #accuracy on test dataset

    auc_acc_list = list()

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    for dataset in datasets:
        knn.fit(dataset[0], dataset[1])
        yHat = knn.predict(xTest)
        yHatProb = knn.predict_proba(xTest)[:,1]
        auc = roc_auc_score(yTest, yHatProb)
        acc = accuracy_score(yTest, yHat)
        auc_acc_list.append([auc, acc])
    
    dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    for dataset in datasets:
        dt.fit(dataset[0], dataset[1])
        yHat = dt.predict(xTest)
        yHatProb = dt.predict_proba(xTest)[:,1]
        auc = roc_auc_score(yTest, yHatProb)
        acc = accuracy_score(yTest, yHat)
        auc_acc_list.append([auc, acc])
    

    perfDF = pd.DataFrame([['KNN All'],
                           ['KNN 5%'],
                           ['KNN 10%'],
                           ['KNN 20%'],
                           ['DT All'],
                           ['DT 5%'],
                           ['DT 10%'],
                           ['DT 20%']],
                           columns=['Dataset'])
    data = pd.DataFrame(auc_acc_list, columns=['AUC', 'Accuracy'])
    final = pd.concat([perfDF, data], axis=1)
    print(final)




if __name__ == "__main__":
    main()