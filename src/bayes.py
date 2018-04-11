# Apply live Naive-Bayes classifier to entries in passed data set
import os
import csv
import copy
import math
from termcolor import colored
# from matplotlib import pyplot


# Global variables
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data/')
feature_list = ['project', 'description', 'tags']

# Convert data .csv file to list of dictionaries structure
def open_csv(file):
    data = []
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)
    return data


# Creates inital empty list of feature values
def setup_beta():
    beta = {}
    
    for feature in feature_list:
        beta.setdefault(feature, {'Unknown': {'True': 0, 'False': 0,
                                  'Total': 0}})
    return beta


# Count new entry of feature information
def update_beta(beta, entry):
    for key, value in entry.items():
        if key not in feature_list:
            break
        elif value not in beta.values():

            # Update Unknown values
            beta[key]['Unknown'][entry['modified']] += 1
            beta[key]['Unknown']['Total'] += 1
            # TODO
            beta[key].setdefault(value, {'True': 1, 'False': 1, 'Total': 1})
            
        # Update new values
        beta[key][value][entry['modified']] += 1
        beta[key][value]['Total'] += 1
    return beta


# Find number of entries that were manually updated
def compute_alpha(beta):
    alpha = [0, 0]

    for feature, categories in beta.items():
        for category in categories:
            alpha[0] += beta[feature][category]['False']
            alpha[1] += beta[feature][category]['True']
    return alpha


# Compute sum of logarithmic beta probabilities for the entry's feature category
def sum_log_ratios(beta, entry):
    log_sum = 0.0

    for feature in feature_list:
        log_sum += math.log(beta[feature][entry[feature]]['True'] /
                            beta[feature][entry[feature]]['False'])
    return log_sum


# Color the larger of two percentages
def color_results(percents, modified):
    results = ['', '', '']
    percent_0 = str(percents[0]) + '% False'
    percent_1 = str(percents[1]) + '% True'

    if modified == 'False':
        results[2] = colored(modified, 'red')
    else:
        results[2] = colored(modified, 'green')

    if percents[0] > percents[1]:
        results[0] = colored(percent_0, 'red')
        results[1] = colored(percent_1, 'white')
    else:
        results[0] = colored(percent_0, 'white')
        results[1] = colored(percent_1, 'green')
    return results


# Determine probability of an entry being manually updated
def compute_probability(beta, theta, output, entry):
    
    # Find logarithmic probability for the given entry, to preserve accuracy
    log_probability = math.log(theta / (1-theta)) + sum_log_ratios(beta, entry)

    # Convert to sigmoid probability, where 0.5 divides false from true
    probability = 1 / (1 + math.e**(-log_probability))

    # Save probability results with the entry
    entry['probability'] = probability

    # Print results if output is requested
    if output:
        p = round(probability * 100, 1)
        percents = [round(100 - p, 1), p]
        results = color_results(percents, entry['modified'])
        print(f"Entry: {entry['project']}, {entry['description']},",
              f"{entry['tags']} - ({results[2]})\n\tProbability:",
              f"{results[1]}, {results[0]}")
    return entry


# Find the total number of misclassifications to show the error rate
def compute_error(errors, count, output, entry):
    prior_rate = errors / ((count - 1) if count > 1 else 1)

    if entry['modified'] == 'False' and entry['probability'] >= 0.5:
        errors += 1     # False positive
    elif entry['modified'] == 'True' and entry['probability'] < 0.5:
        errors += 1     # False negative
    error_rate = errors / count
    entry['error'] = error_rate

    # Print results if output is requested
    result = round(error_rate, 3)
    delta = round(error_rate - prior_rate, 4)

    if delta > 0:
        result = colored(str(result), 'red')
    elif delta < 0:
        result = colored(str(result), 'green')
    elif delta == 0:
        result = colored(str(result), 'white')

    if output:
        print(f"\tError Rate: {result} ({delta})\n")
    
    return (entry, errors)


# Examine time entries, building live Bayesian model
def bayes(skip):
    print('\nBAYES:')

    # Open passed training data set
    with open(os.path.join(data_path, 'train.csv')) as file:
        data_train = open_csv(file)
    print(f"Training model using data from train.csv",
          f"({len(data_train)} entries)")
    
    # Setup list of features to track, termed as beta
    beta = setup_beta()

    # Loop over training data for live model learning
    errors = 0
    print(f"Printing result every {skip} entries, output format is...",
          f"\nEntry: <Project>, <Description>, <Tag> - (<Modified>)",
          f"\n\tProbability: <##.#>% True, <##.#>% False")
    
    for count, entry in enumerate(data_train, 1):
        if data_train.index(entry) % skip == 0:
            output = True
        else:
            output = False

        # Update beta features with new entry
        beta = update_beta(beta, entry)
        
        # Count total number of prior manual updates, termed as alpha
        alpha = compute_alpha(beta)

        # Compute prior probability of an entry being modified, termed as theta
        theta = alpha[1] / (alpha[0] + alpha[1])

        # Calculate probability of manual action based on past values
        entry = compute_probability(beta, theta, output, entry)
        
        # Compute misclassification error rate of model
        entry, errors = compute_error(errors, count, output, entry)
    
    # Graph progression of misclassification error rates
    # pyplot.plot('# of Entries', 'Error Rate', data='error')

    print('Finished training model...')


# DEBUG
if __name__ == '__main__':
    bayes(20)
