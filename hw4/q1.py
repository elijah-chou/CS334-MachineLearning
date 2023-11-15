import argparse
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import random
from math import ceil
from collections import Counter


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    random.shuffle(Lines)
    trainSize = ceil(len(Lines) * 0.7)
    train = Lines[:trainSize]
    test = Lines[trainSize:]
    return train, test


def build_vocab_map(train):
    vocab_map = {}
    for line in train:
        splitLine = line.split()
        checked_words = []
        for wordNum in range(len(splitLine)):
            if wordNum == 0: #skips the first "string" in every line because it is the label of whether it is spam or not
                continue
            else:
                word = splitLine[wordNum]
                if word in vocab_map:
                    if word not in checked_words:
                        vocab_map[word] = vocab_map[word] + 1
                        checked_words.append(word)
                else:
                    vocab_map[word] = 1
                    checked_words.append(word)
    wordsToRemove = []
    for word in vocab_map:
        if vocab_map[word] < 30: #removes words that appear in less than 30 emails from the vocab_map
            wordsToRemove.append(word)
    for word in wordsToRemove:
        del vocab_map[word]
    return vocab_map


def construct_binary(vocab_map, train, test):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    binary_train_x = pd.DataFrame(columns = list(vocab_map.keys()))
    binary_train_y = []
    for line in train:
        row_to_append = []
        splitLine = line.split()
        binary_train_y.append(int(splitLine[0]))
        for word in vocab_map:
            if word in splitLine:
                row_to_append.append(1)
            else:
                row_to_append.append(0)
        binary_train_x.loc[len(binary_train_x)] = row_to_append

    binary_test_x = pd.DataFrame(columns = list(vocab_map.keys()))
    binary_test_y = []
    for line in test:
        row_to_append = []
        splitLine = line.split()
        binary_test_y.append(int(splitLine[0]))
        for word in vocab_map:
            if word in splitLine:
                row_to_append.append(1)
            else:
                row_to_append.append(0)
        binary_test_x.loc[len(binary_test_x)] = row_to_append

    binary_train_y = pd.DataFrame(binary_train_y)
    binary_test_y = pd.DataFrame(binary_test_y)
    return binary_train_x, binary_train_y, binary_test_x, binary_test_y


def construct_count(vocab_map, train, test):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    count_train_x = pd.DataFrame(columns = list(vocab_map.keys()))
    count_train_y = []
    for line in train:
        row_to_append = []
        splitLine = line.split()
        count_train_y.append(int(splitLine[0]))
        c = Counter(splitLine)
        for word in vocab_map:
            if word in splitLine:
                row_to_append.append(c[word])
            else:
                row_to_append.append(0)
        count_train_x.loc[len(count_train_x)] = row_to_append

    count_test_x = pd.DataFrame(columns = list(vocab_map.keys()))
    count_test_y = []
    for line in test:
        row_to_append = []
        splitLine = line.split()
        count_test_y.append(int(splitLine[0]))
        c = Counter(splitLine)
        for word in vocab_map:
            if word in splitLine:
                row_to_append.append(c[word])
            else:
                row_to_append.append(0)
        count_test_x.loc[len(count_test_x)] = row_to_append
    count_train_y = pd.DataFrame(count_train_y)
    count_test_y = pd.DataFrame(count_test_y)
    return count_train_x, count_train_y, count_test_x, count_test_y


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    train, test = model_assessment(args.data)
    vocab_map = build_vocab_map(train)
    binary_train_x, binary_train_y, binary_test_x, binary_test_y = construct_binary(vocab_map, train, test)
    count_train_x, count_train_y, count_test_x, count_test_y = construct_count(vocab_map, train, test)
    binary_train_x.to_csv("binary_train_x.csv")
    binary_train_y.to_csv("binary_train_y.csv")
    binary_test_x.to_csv("binary_test_x.csv")
    binary_test_y.to_csv("binary_test_y.csv")
    count_train_x.to_csv("count_train_x.csv")
    count_train_y.to_csv("count_train_y.csv")
    count_test_x.to_csv("count_test_x.csv")
    count_test_y.to_csv("count_test_y.csv")

if __name__ == "__main__":
    main()
