import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

###########################################################################################

# Decision Tree with Information Gain
class Decision_Tree_Info_Gain():

    def __init__(self, data, original, attributes, label, class_label, prob_limit, minimal_size_split, m):
        # The label column of the data
        self.original_data = original
        self.label_col = label
        self.prob_limit = prob_limit
        self.minimal_size = minimal_size_split
        self.m = m
        # The root of the decision tree
        self.root = self.build_tree(data, attributes, class_label)

    # Get the probability of the class labels
    def get_probability(self, data, class_label):
        label_prob = [0] * len(class_label)
        if len(data) == 0:
            return label_prob

        for row in data:
            for i in range(len(class_label)):
                if row[self.label_col] == class_label[i]:
                    label_prob[i] += 1
                    break
            
        for i in range(len(label_prob)):
            label_prob[i] /= len(data)

        return label_prob
    
    def select_attributes(self, attributes, m):
        selected_attribute = []
        copy_attributes = attributes.copy()

        for i in range(m):
            if not copy_attributes:
                break

            attribute_index = random.randint(0, len(copy_attributes) - 1)
            random_attribute = copy_attributes[attribute_index]
            selected_attribute.append(random_attribute)
            copy_attributes.pop(attribute_index)

        return selected_attribute
    
    # Get entropy of the data
    def entropy(self, data, class_label):
        data_prob = self.get_probability(data, class_label)
        entro_log_sum = 0

        for prob in data_prob:
            if prob == 0:
                return 0
            entro_log_sum += prob * math.log(prob, 2)
        
        return -1 * entro_log_sum

    def info_gain(self, parent_entropy, probabilities, entropies):
        child_entropy = 0
        for prob, entro in zip(probabilities, entropies):
            child_entropy += prob * entro
        return parent_entropy - child_entropy
    
    # Finding the best split among the attributes and the values
    def best_split(self, data, attributes, class_label):
        parent_entropy = self.entropy(data, class_label)
        split_values = []
        attribute_split = None
        data_type = None
        split_branches = []
        max_info_gain = 0

        # Looping through each attribute
        for index, attribute_data in enumerate(attributes):
            attribute = attribute_data[0]
            attribute_type = attribute_data[1]
            attribute_values = set()

            # For numerical attributes
            if attribute_type == 'n':
                data.sort(key = lambda x:x[attribute])
                for i in range(len(data) - 1):
                    value = (data[i][attribute] + data[i + 1][attribute]) / 2
                    attribute_values.add(value)
                
                for attribute_value in attribute_values:
                    branches = [[] for _ in range(2)]
                    left = branches[0]
                    right = branches[1]

                    for data_value in data:
                        if data_value[attribute] <= attribute_value:
                            left.append(data_value)
                        else:
                            right.append(data_value)
                    
                    total_length = len(data)
                    probabilities = [len(left) / total_length, len(right) / total_length]
                    entropies = [self.entropy(left, class_label), self.entropy(right, class_label)]
                    info_gain = self.info_gain(parent_entropy, probabilities, entropies)

                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        split_values = [attribute_value]
                        attribute_split = attribute
                        data_type = 'n'
                        split_branches = branches
            # For categorical attributes
            else:
                for value in self.original_data:
                    attribute_values.add(value[attribute])
                
                attribute_values = list(attribute_values)
            
                branches = [[] for _ in range(len(attribute_values))]
                total_length = len(data)

                for data_value in data:
                    branch_index = attribute_values.index(data_value[attribute])
                    branches[branch_index].append(data_value)
                
                probabilities = []
                entropies = []
                
                for branch in branches:
                    probabilities.append(len(branch) / total_length)
                    entropies.append(self.entropy(branch, class_label))
                
                info_gain = self.info_gain(parent_entropy, probabilities, entropies)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    split_values = attribute_values
                    attribute_split = attribute
                    data_type = 'c'
                    split_branches = branches

        return attribute_split, split_values, data_type, split_branches
    
    def build_tree(self, data, attributes, class_label):
        data_probability = self.get_probability(data, class_label)

        selected_attributes = self.select_attributes(attributes, self.m)
        best_split = self.best_split(data, selected_attributes, class_label)   

        attribute_split = best_split[0]
        split_values = best_split[1]
        attribute_type = best_split[2]
        split_branches = best_split[3]

        if attribute_split is None or any(prob >= self.prob_limit for prob in data_probability) or any(len(branch) == 0 for branch in split_branches) or len(data) <= self.minimal_size:

            predict_label = None
            highest_prob = -1

            for i in range(len(class_label)):
                if data_probability[i] > highest_prob:
                    predict_label = class_label[i]
                    highest_prob = data_probability[i]

            return Leaf(predict_label)

        children = []
        for branch in split_branches:
            children.append(self.build_tree(branch, attributes, class_label))

        return Node(attribute_split, split_values, attribute_type, children)

    # Classify the data
    def classify(self, data):
        return self.root.classify(data)

# This is a node that is used to split the data
class Node():

    def __init__(self, attribute_name, attribute_threshold, attribute_type, children):
        self.attr_name = attribute_name
        self.attr_split = attribute_threshold
        self.attr_type = attribute_type
        self.children = children

    # If less than or equal to then goes to left child, else then goes to right child
    def classify(self, data):
        test_value = data[self.attr_name]
        if self.attr_type == 'n':
            if test_value <= self.attr_split[0]:
                return self.children[0].classify(data)
            else:
                return self.children[1].classify(data)
        else:
            attr_index = self.attr_split.index(test_value)
            return self.children[attr_index].classify(data)

# This is a leaf node that is used to predict the class label of the data
class Leaf():

    def __init__(self, predict_class):
        self.predict_class = predict_class
    
    def classify(self, data):
        return self.predict_class       

# Turn dataframe data into dictionaries inside of a list to easier work with
def csv_to_list(csv_file):
    data = []
    for index, row in csv_file.iterrows():
        row_data = {}
        for key, value in row.items():
            row_data[key] = value
        data.append(row_data)
    return data

def get_class_label(data, label):
    class_label = []

    for value in data:
        if value.get(label) not in class_label:
            class_label.append(value.get(label))

    return class_label

# Test the data using random forest
def test_forest(forest, data, label, class_label):
    true_values = [0] * len(class_label)
    false_values = [0] * len(class_label)

    for value in data:
        majority = None
        highest_count = 0
        class_count = [0] * len(class_label)

        for tree in forest:
            result = tree.classify(value)
            index = class_label.index(result)
            class_count[index] += 1
            if class_count[index] > highest_count:
                highest_count = class_count[index]
                majority = result
        
        majority_index = class_label.index(majority)
        if value[label] == majority:
            true_values[majority_index] += 1
        else:
            false_values[majority_index] += 1

    return true_values, false_values

def bootstrap_data(data): 
    result_data = []
    for i in range(len(data)):
        result_data.append(data[random.randint(0, len(data) - 1)])
    return result_data

def k_fold_stratified(data, k, class_label, label):
    folds = [[] for _ in range(k)]    
    data_group = [[] for _ in range(len(class_label))]

    for data_value in data:
        class_index = class_label.index(data_value[label])
        data_group[class_index].append(data_value)

    for group in data_group:
        fold_size = math.ceil(len(group) / k)
        for i in range(k):
            start_index = i * fold_size
            end_index = start_index + fold_size
            folds[i].extend(group[start_index : end_index])

    return folds

def random_forest(ntree_values, data, attributes, label, class_label, gini):
    k_folds = k_fold_stratified(data, 10, class_label, label)

    mean_accuracy = []
    mean_precision = []
    mean_recall = []
    mean_f1 = []

    accuracy = [[] for _ in range(len(ntree_values))]
    precision = [[] for _ in range(len(ntree_values))]
    recall = [[] for _ in range(len(ntree_values))]
    f1 = [[] for _ in range(len(ntree_values))]
    beta = 1
    beta_square = pow(beta, 2)

    for i in range(len(k_folds)):
        training = k_folds[:i] + k_folds[i + 1:]
        training = [element for sub_array in training for element in sub_array]
        minimal_size_split = 0.05 * len(training)
        testing = k_folds[i]

        for j in range(len(ntree_values)):
            ntree = ntree_values[j]
            forest = []

            for _ in range(ntree):
                boostrap_training = bootstrap_data(training)
                if not gini:
                    tree = Decision_Tree_Info_Gain(boostrap_training, training, attributes, label, class_label, 1, minimal_size_split, int(math.sqrt(len(attributes))))
                else:
                    tree = Decision_Tree_Gini(boostrap_training, training, attributes, label, class_label, 1, minimal_size_split, int(math.sqrt(len(attributes))))
                forest.append(tree)
            
            true_values, false_values = test_forest(forest, testing, label, class_label)
            total_size = len(testing)

            if len(class_label) == 2:
                true_pos = true_values[0]
                true_neg = true_values[1]
                false_pos = false_values[0]
                false_neg = false_values[1]

                ntree_accuracy = (true_pos + true_neg) / total_size
                ntree_precision = true_pos / (true_pos + false_pos)
                ntree_recall = true_pos / (true_pos + false_neg)
                ntree_f1 = (1 + beta_square) * (ntree_precision * ntree_recall) / ((beta_square * ntree_precision) + ntree_recall)
            else:
                ntree_accuracy = sum(true_values) / total_size
                ntree_precision = 0
                ntree_recall = 0
                false_sum = sum(false_values)

                for n in range(len(class_label)):
                    if true_values[n] == 0:
                        ntree_precision += 0
                        ntree_recall += 0
                    else:
                        ntree_precision += true_values[n] / (true_values[n] + false_values[n])
                        ntree_recall += true_values[n] / (true_values[n] + (false_sum - false_values[n]))

                ntree_precision /= len(class_label)
                ntree_recall /= len(class_label)

                ntree_f1 = (1 + beta_square) * (ntree_precision * ntree_recall) / ((beta_square * ntree_precision) + ntree_recall)

            accuracy[j].append(ntree_accuracy)
            precision[j].append(ntree_precision)
            recall[j].append(ntree_recall)
            f1[j].append(ntree_f1)
    
    for i in range(len(ntree_values)):
        mean_accuracy.append(np.mean(accuracy[i]))
        mean_precision.append(np.mean(precision[i]))
        mean_recall.append(np.mean(recall[i]))
        mean_f1.append(np.mean(f1[i]))
    
    return mean_accuracy, mean_precision, mean_recall, mean_f1

###########################################################################################
ntree_values = [1, 5, 10, 20, 30, 40, 50]
###########################################################################################
