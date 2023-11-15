from numpy.random.mtrand import normal
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def main():
    scaler = StandardScaler()
    xTrain = pd.read_csv("q4xTrain.csv")
    xTest = pd.read_csv("q4xTest.csv")
    yTrain = pd.read_csv("q4yTrain.csv")
    yTrain = yTrain.to_numpy()
    yTrain = np.ravel(yTrain)
    yTest = pd.read_csv("q4yTest.csv")
    yTest = yTest.to_numpy()
    yTest = np.ravel(yTest)
    xTrain_scaled = scaler.fit_transform(xTrain)
    xTest_scaled = scaler.transform(xTest)
    
    
    clf = LogisticRegression(penalty='none').fit(xTrain_scaled, yTrain)
    normal_probs = clf.predict_proba(xTest_scaled)
    normal_probs = normal_probs[:, 1]
    normal_fpr, normal_tpr, _ = roc_curve(yTest, normal_probs)
    normal_auc = roc_auc_score(yTest, normal_probs)

    pca = PCA(n_components=8)
    xTrain_pca = pca.fit_transform(xTrain_scaled)
    print(pca.explained_variance_ratio_)
    print(pca.components_)
    clf = LogisticRegression(penalty='none').fit(xTrain_pca, yTrain)
    xTest_pca = pca.transform(xTest_scaled)
    pca_probs = clf.predict_proba(xTest_pca)
    pca_probs = pca_probs[:, 1]
    pca_fpr, pca_tpr, _ = roc_curve(yTest, pca_probs)
    pca_auc = roc_auc_score(yTest, pca_probs)

    #code for generating the ROC curve plot
    pyplot.plot(normal_fpr, normal_tpr, label="Normalized")
    pyplot.plot(pca_fpr, pca_tpr, label="PCA")
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.legend()
    pyplot.show()

    print("Normalized: ROC AUC = %.3f" % (normal_auc))
    print("PCA: ROC AUC = %.3f" % (pca_auc))


    

if __name__ == "__main__":
    main()