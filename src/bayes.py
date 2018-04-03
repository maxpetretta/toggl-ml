# Apply Naive-Bayes classifier to entries in passed data set
import os
import csv
import copy
from preprocess import preprocess


# Global variables
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data/')


# Find number of entries that were manually updated
def compute_alpha(data):
    alpha_0 = 0
    alpha_1 = 0

    for row in data:
        if row['modified'] == 'True':
            alpha_1 += 1
        else:
            alpha_0 += 1
    return (alpha_0, alpha_1)


# Count occurances of a feature when an entry is modified
def compute_beta(data, key, item):
    beta = {}
    total = 0
    modified = 0
    
    for row in data:
        if row[key] == item:
            total += 1
            if row['modified'] == 'True':
                modified += 1
    beta['True'] = modified
    beta['False'] = total - modified
    beta['Total'] = total

    print(key, item, beta)
    return beta


# Examine time entries, building Bayesian model
def bayes(file_name, categories):

    # Open passed data set
    with open(os.path.join(data_path, file_name)) as file:
        data = []
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    # Count total number of manual updates, termed as alpha
    alpha_0, alpha_1 = compute_alpha(data)

    # Compute overall probability of an entry being modified, termed as theta
    theta = alpha_1 / (alpha_0 + alpha_1)
    
    # Build feature probability lookup table, termed as beta
    beta = copy.deepcopy(categories)
    for key, value in categories.items():
        for item in value:
            beta[key][item] = compute_beta(data, key, item)
    print(beta)



# DEBUG
if __name__ == '__main__':
    bayes('train.csv', preprocess(0.4, 0.3, 0.3))
