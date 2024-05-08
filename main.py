from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib . pyplot as plt

from michael_ML import neural_network, k_NN, random_forest

digits_dataset = datasets.load_digits(return_X_y=True)
digits_feature_data = digits_dataset[0]
digits_class_data = digits_dataset[1]
digits_class_col = 'digit'

digits_feature_df = pd.DataFrame(digits_feature_data)
digits_feature_type = ['n' for _ in range(len(digits_feature_data[0]))]
digits_feature_type = list(zip(digits_feature_df.columns, digits_feature_type))

digits_class_df = pd.DataFrame(digits_class_data, columns=['digit'])
#####################################################################################
titanic_file = './data/titanic.csv'
titanic_df = pd.read_csv(titanic_file)
titanic_class_col = 'Survived'
titanic_features_to_drop = ['Name']

titanic_feature_data = titanic_df.drop(columns=[titanic_class_col])
titanic_feature_data = titanic_feature_data.drop(columns=titanic_features_to_drop)

titanic_feature_type = ['c', 'c', 'n', 'n', 'n', 'n']
titanic_feature_type = list(zip(titanic_feature_data.columns, titanic_feature_type))

titanic_class_data = titanic_df[[titanic_class_col]]
#####################################################################################
loan_file = './data/loan.csv'
loan_df = pd.read_csv(loan_file)
loan_class_col = 'Loan_Status'
loan_features_to_drop = ['Loan_ID']

loan_feature_data = loan_df.drop(columns=[loan_class_col])
loan_feature_data = loan_feature_data.drop(columns=loan_features_to_drop)

loan_feature_type = ['c', 'c', 'c', 'c', 'c', 'n', 'n', 'n', 'n', 'c', 'c']
loan_feature_type = list(zip(loan_feature_data.columns, loan_feature_type))

loan_class_data = loan_df[[loan_class_col]]

# k_NN.K_NN('loan k_NN', loan_feature_data, loan_feature_type, loan_class_data, loan_class_col)
# neural_network.Neural_Network('loan neural network', loan_feature_data, loan_feature_type, loan_class_data, loan_class_col)
#####################################################################################
parkinson_file = './data/parkinsons.csv'
parkinson_df = pd.read_csv(parkinson_file)
parkinson_class_col = 'Diagnosis'

parkinson_feature_data = parkinson_df.drop(columns=[parkinson_class_col])

parkinson_feature_type = ['n' for _ in range(len(parkinson_feature_data.columns))]
parkinson_feature_type = list(zip(parkinson_feature_data.columns, parkinson_feature_type))

parkinson_class_data = parkinson_df[[parkinson_class_col]]