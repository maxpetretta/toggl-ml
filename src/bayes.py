# Apply Naive-Bayes classifier to entries in passed data set
import os
import csv
import copy
import math
from termcolor import colored
from preprocess import preprocess


# Global variables
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data/')

# Convert data .csv file to list of dictionaries structure
def open_csv(file):
    data = []
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)
    return data


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
    beta['Total'] = (total if total > 0 else 1) # TODO
    return beta


# Determine probability of an entry being manually updated
def compute_probability(theta, beta, data, lines):
    print('Output format is...\nEntry: <Project>, <Description>, <Tag>',
          '- (<Modified>)\n\tProbability: <##.#>% True, <##.#>% False\n')

    for row in data:
        probability_0 = beta_product(beta, row, 'False')
        probability_1 = beta_product(beta, row, 'True')

        probability_0 *= ((theta**0) * (1 - theta)**1)
        probability_1 *= ((theta**1) * (1 - theta)**0)

        if probability_0 + probability_1 > 0 and probability_0 > probability_1:
            percentage_0 = colored(str(round((probability_0 / (probability_0
                + probability_1)) * 100, 1)) + '% False', 'red')
            percentage_1 = colored(str(round((probability_1 / (probability_0
                + probability_1)) * 100, 1)) + '% True', 'white')
        elif probability_0 + probability_1 > 0 and probability_1 > probability_0:
            percentage_0 = colored(str(round((probability_0 / (probability_0
                + probability_1)) * 100, 1)) + '% False', 'white')
            percentage_1 = colored(str(round((probability_1 / (probability_0
                + probability_1)) * 100, 1)) + '% True', 'green')
        else:
            percentage_0 = colored(str(round(probability_0, 1)) + '% False', 'white')
            percentage_1 = colored(str(round(probability_1, 1)) + '% True', 'white')

        if lines > 0:
            print(f"Entry: {row['project']}, {row['description']},",
                  f"{row['tags']} - ({row['modified']})\n\tProbability:",
                  f"{percentage_1}, {percentage_0}")
            lines -= 1


# Determine probability of an entry being manually updated
def compute_log_probability(theta, beta, data, lines):
    print('Output format is...\nEntry: <Project>, <Description>, <Tag>',
          '- (<Modified>)\n\tProbability: <##.#>% True, <##.#>% False\n')

    for row in data:
        log_probability_ratio = log_beta_sum(beta, row)

        log_probability_ratio += math.log(theta / (1 - theta))

        if probability_0 + probability_1 > 0 and probability_0 > probability_1:
            percentage_0 = colored(str(round((probability_0 / (probability_0
                + probability_1)) * 100, 1)) + '% False', 'red')
            percentage_1 = colored(str(round((probability_1 / (probability_0
                + probability_1)) * 100, 1)) + '% True', 'white')
        elif probability_0 + probability_1 > 0 and probability_1 > probability_0:
            percentage_0 = colored(str(round((probability_0 / (probability_0
                + probability_1)) * 100, 1)) + '% False', 'white')
            percentage_1 = colored(str(round((probability_1 / (probability_0
                + probability_1)) * 100, 1)) + '% True', 'green')
        else:
            percentage_0 = colored(str(round(probability_0, 1)) + '% False', 'white')
            percentage_1 = colored(str(round(probability_1, 1)) + '% True', 'white')

        if lines > 0:
            print(f"Entry: {row['project']}, {row['description']},",
                  f"{row['tags']} - ({row['modified']})\n\tProbability:",
                  f"{percentage_1}, {percentage_0}")
            lines -= 1


# Compute product of beta probabilities for given feature
def beta_product(beta, row, outcome):
    probability = 1.0
    features = ['project', 'description', 'tags']

    for feature in features:
        probability *= (beta[feature][row[feature]][outcome] /
                        beta[feature][row[feature]]['Total'])
    return probability


# Compute product of beta probabilities for given feature
def log_beta_sum(beta, row):
    log_probability = 0.0
    features = ['project', 'description', 'tags']

    for feature in features:
        log_probability += math.log(beta[feature][row[feature]]['True'] /
                        beta[feature][row[feature]]['False'])
    return log_probability


# Examine time entries, building Bayesian model
def bayes(features, lines):
    print('BAYES:')

    # Open passed training data set
    with open(os.path.join(data_path, 'train.csv')) as file:
        data_train = open_csv(file)
    print(f"Training model using data from train.csv",
          f"({len(data_train)} entries)")
    
    # Count total number of prior manual updates, termed as alpha
    alpha_0, alpha_1 = compute_alpha(data_train)

    # Compute prior probability of an entry being modified, termed as theta
    theta = alpha_1 / (alpha_0 + alpha_1)
    
    # Build evidence likelihood lookup table, termed as beta
    beta = copy.deepcopy(features)
    for key, value in features.items():
        for item in value:
            beta[key][item] = compute_beta(data_train, key, item)

    # Open passed testing data set
    with open(os.path.join(data_path, 'test.csv')) as file:
        data_test = open_csv(file)
    print(f"Testing model using data from test.csv",
          f"({len(data_test)} entries)")
    
    # Calculate probability of manual action on test data set
    compute_probability(theta, beta, data_test, lines)
    print('Finished testing model...\n')


# DEBUG
if __name__ == '__main__':
    features = preprocess(0.5, 0.2, 0.3)
    bayes(features, 20)
