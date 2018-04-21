# Apply live probability classifiers to entries in passed data set
import os
import csv
import math
import dateutil.parser
from scipy import special
from termcolor import colored


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


# Examine training data to find value for hyperparameter kappa
def compute_kappa(data, target):

    # Find mean of target time feature
    mean = 0
    hours = []
    for entry in data:
        hour = (dateutil.parser.parse(entry[target]).hour / 24) * (2*math.pi)   # TODO
        mean += hour
        hours.append(hour)
    mean /= len(hours)

    # Calculate variance from list of time hours
    variance = 0
    for hour in hours:
        variance += (hour - mean)**2
    variance /= len(hours)

    kappa = 1 / (2*variance)
    return kappa


# Creates inital empty list of feature values
def setup_beta():
    beta = {}
    
    for feature in feature_list:
        beta.setdefault(feature, {'Unknown':
            {'True': 0, 'False': 0, 'Total': 0}})
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
            beta[key].setdefault(value, {'True': 1, 'False': 1, 'Total': 1})
            
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


# Determine probabilities for categorical features on time entry
def compute_prob_categorical(entry, previous):

    # Update previous beta features with new entry
    beta = (previous['beta'] if previous is not None else setup_beta())
    beta = update_beta(entry, beta)
    
    # Count total number of prior manual updates, termed as alpha
    alpha = compute_alpha(beta)

    # Compute prior probability of an entry being modified, termed as theta
    theta = alpha[1] / (alpha[0] + alpha[1])
    
    # Find logarithmic probability for the categorical features and save values
    prob_categorical = math.log(theta / (1-theta)) + sum_log_ratios(entry, beta)
    entry['beta'] = beta

    return (entry, prob_categorical)


# Compute new values of the von Mises concentration and direction parameters
def update_hyperparameters(x, a, b, kappa):
    a_prime = kappa * ((a * math.sin(b)) + math.sin(x))
    
    b_numerator = (a * math.sin(b)) + math.sin(x)
    b_denominator = (a * math.cos(b)) + math.cos(x)
    b_prime = math.atan(b_numerator / b_denominator)

    # Enforce parameter limits
    a_prime = (a_prime if a_prime > 0 else 0.001)

    b_prime = (2 * b_prime) + math.pi
    b_prime = (b_prime if b_prime > 0 and b_prime <= (2*math.pi) else 0.001)
    
    return (a_prime, b_prime)


# Determine probabilities for target time features on time entry
def compute_prob_time(entry, previous, kappa, target):

    # Retrieve previous hyperparameter values, a & b
    if previous is not None:
        a_0, a_1 = previous['a_0'], previous['a_1']
        b_0, b_1 = previous['b_0'], previous['b_1']
    else:
        a_0 = a_1 = math.pi
        b_0 = b_1 = 1 / (2*(math.pi**2))

    # Update von Mises hyperparameters with the new value of target time x
    x = (dateutil.parser.parse(entry[target]).hour / 24) * (2*math.pi)
    if entry['modified'] == 'False':
        a_0, b_0 = update_hyperparameters(x, a_0, b_0, kappa)
    else:
        a_1, b_1 = update_hyperparameters(x, a_1, b_1, kappa)
    
    # Find logaritmic Bessel function ratio
    bessel = math.log(special.iv(0, a_0) / special.iv(0, a_1))

    # Compute final time probability and save values
    prob_time = ((a_1 * math.cos(x-b_1)) - (a_0 * math.cos(x-b_0))) + bessel
    entry['a_0'], entry['a_1'] = a_0, a_1
    entry['b_0'], entry['b_1'] = b_0, b_1

    return (entry, prob_time)


# Determine probability for duration feature on time entry
def compute_prob_duration(entry, previous):

    # Retrieve previous hyperparameter values, c & d
    if previous is not None:
        c_0, c_1 = previous['c_0'], previous['c_1']
        d_0, d_1 = previous['d_0'], previous['d_1']
    else:
        c_0 = c_1 = 1
        d_0 = d_1 = 1
    
    # Update gamma hyperparameters with new value of duration x
    x = int(entry['duration']) / 3600000 / 24 / 7
    if entry['modified'] == 'False':
        c_0, d_0 = c_0 + 1, (d_0 / (1 + d_0*x)) 
    else:
        c_1, d_1 = c_1 + 1, (d_1 / (1 + d_1*x))
    
    # # Find logarithmic gamma function ratio
    # gamma_ratio = math.lgamma(c_0+1) - math.lgamma(c_1+1)

    # # Compute final duration probability and save values
    # term_c = (c_1 - c_0) * math.log((x if x > 0 else 1))
    # term_d = (d_0 - d_1) * x
    # term_cd = (c_1 * math.log(d_1)) - (c_0 * math.log(d_0))
    
    # prob_duration = term_c + term_d + term_cd + gamma_ratio

    # Log of the ratio of the probability of duration
    term_cd = (d_0*c_1 - d_1*c_0) / (d_0*d_1)
    term_log_cd = math.log(c_1/c_0) + math.log(d_0 / d_1)

    prob_duration = term_cd * (-x) * term_log_cd
    entry['c_0'], entry['d_0'] = c_0, d_0
    entry['c_1'], entry['d_1'] = c_1, d_1

    return (entry, prob_duration)


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


# Determine probability of the entry being modified from feature probabilities
def compute_prob_sigmoid(entry, p1, p2, p3, p4, output):
    
    # Sum all probability values
    prob_sum = p1 + p2 + p3 + p4
    
    # Convert to sigmoid probability, where 0.5 divides false from true
    probability = 1 / (1 + math.e**(-prob_sum))
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
    prior_error_rate = (previous['error'] if previous is not None else 0)
    prior_entropy_loss = (previous['entropy'] if previous is not None else 0)

    entropy_loss = 0
    log_ratio = math.log((1 - entry['probability']) / entry['probability'])

    if entry['modified'] == 'False' and entry['probability'] >= 0.5:
        errors += 1     # False positive
        entropy_loss -= log_ratio
    elif entry['modified'] == 'True' and entry['probability'] < 0.5:
        errors += 1     # False negative
        entropy_loss += log_ratio
    
    # Find new mean error rate and entropy loss rate
    ratio = 1 / count
    error_rate = ratio * errors # (1-ratio) * prior_rate) + (ratio * error_rate)
    entry['error'] = error_rate

    entropy_rate = ((1-ratio) * prior_entropy_loss) + (ratio * entropy_loss)
    entry['entropy'] = entropy_rate

    # Print results if output is requested
    if output:
        delta = round(error_rate - prior_error_rate, 4)
        delta = "{0:+}".format(delta)
        rounded_rate = round(error_rate, 3)
        print(f"\tError Rate: {rounded_rate} ({delta})\n")

    return (entry, errors)


# Examine time entries, building live Bayesian model
def learn(skip):
    print('\nBAYES:')

    # Open training data set
    with open(os.path.join(data_path, 'train.csv')) as file:
        data_train = open_csv(file)
    print(f"Training model using data from train.csv",
          f"({len(data_train)} entries)")

    # Loop over training data for live model learning
    errors = 0
    kappa_start = compute_kappa(data_train, 'start')
    kappa_end = compute_kappa(data_train, 'end')
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
        entry, prob_time_start = compute_prob_time(entry, previous,
                                                   kappa_start, 'start')
        entry, prob_time_end = compute_prob_time(entry, previous,
                                                 kappa_end, 'end')

        # Calculate probability of manual action based on duration value
        entry, prob_duration = compute_prob_duration(entry, previous)

        # Calculate true probability using sigmoid function
        entry = compute_prob_sigmoid(entry, prob_categorical, prob_time_start,
                                     prob_time_end, prob_duration, output)
        
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
    learn(50)
