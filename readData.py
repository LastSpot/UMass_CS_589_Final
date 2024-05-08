from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib . pyplot as plt

from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib . pyplot as plt

def readData(name, numFolds):
    if name == 'titanic':
        data = pd.read_csv('./data/titanic.csv', delimiter = ',')
        data = data.sample(frac=1).values
        # X is all the attributes and y is the label, you can print to have a better understanding here
        X = data[:, 1:]
        Y = data[:, 0]
        first_layer = 7
        final_layer = 2
        # final label - did not survive = 0, survived = 1
        labels = [0,1]
    if name == 'loan':
        data = pd.read_csv('./data/loan.csv', delimiter = ',')
        data = data.sample(frac=1).values
        # X is all the attrs and y is the label
        X = data[:, :12]
        Y = data[:, 12]
        first_layer = 12
        final_layer = 2
        # final label - No = 0, yes = 1
        labels = [0,1]
    if name == 'parkinsons':
        data = pd.read_csv('./data')
        parkinson_file = './data/parkinsons.csv'
        parkinson_df = pd.read_csv(parkinson_file)

readData("loan", 10)