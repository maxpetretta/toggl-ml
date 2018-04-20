# Apply live Naive-Bayes classifier to entries in passed data set
import os
import csv
import copy
import math
import dateutil.parser
from termcolor import colored
from scipy import special


# Global variables
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data/')
feature_list = ['project', 'description', 'tags']

kappa = (2/24) * (2*math.pi)


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
def update_beta(entry, beta):
    for key, value in entry.items():
        if key not in feature_list:
            break
        elif value not in beta.values():

            # New values add to the 'Unknown' category
            beta[key]['Unknown'][entry['modified']] += 1
            beta[key]['Unknown']['Total'] += 1
            beta[key].setdefault(value, {'True': 1, 'False': 1, 'Total': 1})    # TODO
            
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
def sum_log_ratios(entry, beta):
    log_sum = 0.0

    for feature in feature_list:
        log_sum += math.log(beta[feature][entry[feature]]['True'] /
                            beta[feature][entry[feature]]['False'])
    return log_sum


# Color the larger of two percentages
def color_output(percents, modified):
    results = ['', '', '']
    percent_true = str(percents[1]) + '% True'
    percent_false = str(percents[0]) + '% False'
    
    color_true = ('green' if modified == 'True' else 'white')
    color_false = ('white' if modified == 'True' else 'red')
    color_result = ('green' if modified == 'True' else 'red')

    results[0] = colored(percent_false, color_false)
    results[1] = colored(percent_true, color_true)
    results[2] = colored(modified, color_result)
    return results


# Determine probabilities for categorical features on time entry
def compute_prob_categorical(entry, previous):

    # Update previous beta features with new entry
    old_beta = (previous['beta'] if previous is not None else setup_beta())
    beta = update_beta(entry, old_beta)
    
    # Count total number of prior manual updates, termed as alpha
    alpha = compute_alpha(beta)

    # Compute prior probability of an entry being modified, termed as theta
    theta = alpha[1] / (alpha[0] + alpha[1])
    
    # Find logarithmic probability for the categorical features and save values
    prob_categorical = math.log(theta / (1-theta)) + sum_log_ratios(entry, beta)
    entry['beta'] = beta

    return (entry, prob_categorical)


# Compute new values of the von Mises concentration and direction parameters
def update_hyperparameters(x, a, b):
    a_prime = kappa * ((a * math.sin(b)) + math.sin(x))
    
    b_numerator = (a * math.sin(b)) + math.sin(x)
    b_denominator = (a * math.cos(b)) + math.cos(x)
    b_prime = math.atan(b_numerator / b_denominator)

    # Enforce parameter limits
    a_prime = (a_prime if a_prime > 0 else (0.01 * math.pi))

    b_prime = (2 * b_prime) + math.pi
    b_prime = (b_prime if b_prime > 0 and b_prime <= (2*math.pi) else (0.01*math.pi))
    
    # print('a_prime: ', a_prime)
    # print('b_prime: ', b_prime)
    return (a_prime, b_prime)


# Determine probabilities for target time features on time entry
def compute_prob_time(entry, previous, target):

    # Retrieve previous hyperparameter values
    if previous is not None:
        a_0, a_1, b_0, b_1 = (previous['a_0'], previous['a_1'],
            previous['b_0'], previous['b_1'])
    else:
        a_0, a_1, b_0, b_1 = (math.pi, math.pi,
            1 / (2*(math.pi**2)), 1 / (2*(math.pi**2)))

    # Update von Mises hyperparameters with the new value of target time x
    x = (dateutil.parser.parse(entry[target]).hour / 24) * (2*math.pi)
    a_0_prime, b_0_prime = (update_hyperparameters(x, a_0, b_0)
        if entry['modified'] == 'False' else (a_0, b_0))
    a_1_prime, b_1_prime = (update_hyperparameters(x, a_1, b_1)
        if entry['modified'] == 'True' else (a_1, b_1))
    
    # Find logaritmic Bessel function ratio
    bessel = math.log(special.iv(0, a_0_prime) / special.iv(0, a_1_prime))

    # Compute final time probability and save values
    prob_time = ((a_1_prime * math.cos(x-b_1_prime))
        - (a_0_prime * math.cos(x-b_0_prime))) + bessel
    entry['a_0'], entry['b_0'] = a_0_prime, b_0_prime
    entry['a_1'], entry['b_1'] = a_1_prime, b_1_prime

    return (entry, prob_time)


# Determine probability for duration feature on time entry
def compute_prob_duration(entry, previous):

    # Retrieve previous hyperparameter values
    alpha_0, beta_0 = ((previous['alpha_0'], previous['beta_0'])
        if previous is not None else (1, 1))                                    # TODO
    alpha_1, beta_1 = ((previous['alpha_1'], previous['beta_1'])
        if previous is not None else (1, 1))                                    # TODO
    
    # Update gamma hyperparameters with new value of duration x
    x = int(entry['duration'])
    alpha_0, beta_0 = ((alpha_0 + 1, beta_0 + x)
        if entry['modified'] == 'False' else (alpha_0, beta_0))
    alpha_1, beta_1 = ((alpha_1 + 1, beta_1 + x)
        if entry['modified'] == 'True' else (alpha_1, beta_1))
    
    # Find logarithmic gamma function ratio
    gamma_ratio = math.log(special.gamma(alpha_0) / special.gamma(alpha_1))     # TODO

    # Compute final duration probability and save values
    term_alpha = (alpha_1 - alpha_0) * math.log((x if x > 0 else 1))
    term_beta = (beta_0 - beta_1) * x
    term_combined = (alpha_1 * math.log(beta_1)) - (alpha_0 * math.log(beta_0))
    
    prob_duration = term_alpha + term_beta + term_combined + gamma_ratio
    entry['alpha_0'], entry['beta_0'] = alpha_0, beta_0
    entry['alpha_1'], entry['beta_1'] = alpha_1, beta_1

    return (entry, prob_duration)


# Determine probability of the entry being modified from feature probabilities
def compute_prob_sigmoid(entry, p1, p2, p3, p4, output):
    
    # Sum probability values
    prob_sum = p1 + p2 + p3 + p4
    
    # Convert to sigmoid probability, where 0.5 divides false from true
    probability = 1 / (1 + math.e**(-prob_sum))

    # Save probability results with the entry
    entry['probability'] = probability

    # Print results if output is requested
    if output:
        p = round(probability * 100, 1)
        percents = [round(100 - p, 1), p]
        results = color_output(percents, entry['modified'])
        print(f"Entry: {entry['project']}, {entry['description']},",
            f"{entry['tags']} - ({results[2]})\n\tProbability:",
            f"{results[1]}, {results[0]}")
    return entry
    

# Find the total number of misclassifications to show the mean error rate
def compute_error(entry, previous, errors, count, output):

    # Retrieve prior error rate for calculating the mean
    prior_rate = (previous['error'] if previous is not None else 0)

    if entry['modified'] == 'False' and entry['probability'] >= 0.5:
        errors += 1     # False positive
    elif entry['modified'] == 'True' and entry['probability'] < 0.5:
        errors += 1     # False negative
    
    # Find new mean error rate based on the previously examined entries
    mean_split = 1 / count
    error_rate = mean_split * errors
    mean_rate = ((1-mean_split) * prior_rate) + (mean_split * error_rate)
    entry['error'] = mean_rate

    # Print results if output is requested
    if output:
        delta = round(mean_rate - prior_rate, 4)
        delta = "{0:+}".format(delta)
        rounded_rate = round(mean_rate, 3)
        print(f"\tError Rate: {rounded_rate} ({delta})\n")

    return (entry, errors)


# Examine time entries, building live Bayesian model
def bayes(skip):
    print('\nBAYES:')

    # Open training data set
    with open(os.path.join(data_path, 'train.csv')) as file:
        data_train = open_csv(file)
    print(f"Training model using data from train.csv",
          f"({len(data_train)} entries)")

    # Loop over training data for live model learning
    errors = 0
    print(f"Printing result every {skip} entries, output format is...",
          f"\nEntry: <Project>, <Description>, <Tag> - (<Modified>)",
          f"\n\tProbability: <##.#>% True, <##.#>% False",
          f"\n\tError Rate: <#.###> (<+/- ####>)\n")
    
    for count, entry in enumerate(data_train, 1):
        output = (True if count % skip == 0 else False)
        previous = (data_train[count - 2] if count > 1 else None)

        # Calculate probability of manual action based on categorical values
        entry, prob_categorical = compute_prob_categorical(entry, previous)

        # Calculate probabilities of manual action based on time values
        entry, prob_time_start = compute_prob_time(entry, previous, 'start')
        entry, prob_time_end = compute_prob_time(entry, previous, 'end')

        # Calculate probability of manual action based on duration value
        # entry, prob_duration = compute_prob_duration(entry, previous)

        # Calculate true probability using sigmoid function
        entry = compute_prob_sigmoid(entry, prob_categorical, prob_time_start,
                                     prob_time_end, 0, output)
        
        # Compute misclassification error rate of model
        entry, errors = compute_error(entry, previous, errors, count, output)
    
    # Save updated training data to new .csv file
    keys = data_train[0].keys()
    with open(os.path.join(data_path, 'model.csv'), 'w') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_train)
    print('Finished training model...')


# DEBUG
if __name__ == '__main__':
    bayes(20)
