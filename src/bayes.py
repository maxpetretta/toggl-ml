# Apply Naive-Bayes classifier to entries in passed data set
import os
import csv
from preprocess import preprocess


# Global variables
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data/')


# Find number of entries that were manually updated
def compute_alpha(data):
    alpha_0 = 0
    alpha_1 = 0

    for row in data:
        print(row['modified'])
        if row['modified'] == 'True':
            alpha_1 += 1
        else:
            alpha_0 += 1
    return (alpha_0, alpha_1)


# Count occurances of a feature when an entry is modified
def compute_beta(data, categories):
    


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
    
    # Build feature lookup probability table, termed as beta
    beta = compute_beta(data, categories)




# DEBUG
if __name__ == '__main__':
    bayes('train.csv', preprocess(0.4, 0.3, 0.3))
