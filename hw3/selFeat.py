import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    datetime = pd.to_datetime(df["date"])
    df["datetime"] = datetime
    df['day'] = df["datetime"].dt.day
    df = df.drop(columns=['date', 'datetime'])
    return df


def select_features(df, dropList):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    if dropList is None:
        correlations = df.corr()
        featDropIndex = []
        for column in range(correlations.shape[1]):
            for row in range(correlations.shape[0]):
                if column in featDropIndex:
                    break
                if column >= row:
                    continue
                if abs(correlations.iloc[row, column]) > 0.5:
                    featDropIndex.append(row)
        finalList = list(set(featDropIndex))
        finalList.sort(reverse=True)
        for feature in finalList:
            df = df.drop(df.columns[feature], axis=1)
        return df, finalList
    else:
        for feature in dropList:
            df = df.drop(df.columns[feature], axis=1)
        return df, dropList


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    scaler = StandardScaler()
    scaledTrainDF = scaler.fit_transform(trainDF)
    scaledTestDF = scaler.transform(testDF)
    xTrainFinal = pd.DataFrame(scaledTrainDF)
    xTestFinal = pd.DataFrame(scaledTestDF)
    return xTrainFinal, xTestFinal


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    dropFeatList = None
    xNewTrain, dropFeatList = select_features(xNewTrain, dropFeatList)
    xNewTest, dropFeatList = select_features(xNewTest, dropFeatList)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
