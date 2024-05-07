from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib . pyplot as plt

digits_dataset = datasets.load_digits(return_X_y=True)
digits_features = digits_dataset[0]
digits_labels = digits_dataset[1]

titanic_file = './data/titanic.csv'
titanic_df = pd.read_csv(titanic_file)
titanic_features = titanic_df.drop(columns=['Survived'])

loan_file = './data/loan.csv'
load_df = pd.read_csv(loan_file)

parkinson_file = './data/parkinsons.csv'
parkinson_df = pd.read_csv(parkinson_file)