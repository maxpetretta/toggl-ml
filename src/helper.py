# Contains helper functions for the toggl-ml project
import csv
import math
import random
import dateutil.parser


# Global variables
feature_list = ['project', 'description', 'tags']


# UNIVERSAL HELPERS

# Convert data .csv file to list of dictionaries structure
def open_csv(file):
    data = []
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)
    return data


# Calculate the F1 and F2 scores for the given entry values
def compute_scores(bundle):
    scores = [0, 0]
    values = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    for entry in bundle:
        modified = entry['modified']
        probability = float(entry['probability'])

        if modified == 'True' and probability >= 0.5:
            values['tp'] += 1
        elif modified == 'False' and probability < 0.5:
            values['tn'] += 1
        elif modified == 'False' and probability >= 0.5:
            values['fp'] += 1
        elif modified == 'True' and probability < 0.5:
            values['fn'] += 1
    
    # Calculate F1 score, for modified values
    tp, tn, fp, fn = values['tp'], values['tn'], values['fp'], values['fn']
    precision = 1 / (1 + (fp/(tp if tp > 0 else 1)))
    recall = 1 / (1 + (fn/(tp if tp > 0 else 1)))
    f1_score = 2 / ((1/precision) + (1/recall))
    scores[0] = f1_score

    # Calculate F2 score, for non-modified values
    precision = 1 / (1 + (fn/(tn if tn > 0 else 1)))
    recall = 1 / (1 + (fp/(tn if tn > 0 else 1)))
    f2_score = 2 / ((1/precision) + (1/recall))
    scores[1] = f2_score

    return ([tp, tn, fp, fn], scores)


# LEARN HELPERS

# Examine distribution of time data to find value for hyperparameter kappa
def compute_kappa(data, target):

    # Find mean of target time feature
    mean = 0
    hours = []
    for entry in data:
        hour = (dateutil.parser.parse(entry[target]).hour / 24) * (2*math.pi)   
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


# Creates inital empty list of feature category values
def setup_beta():
    beta = {}
    
    for feature in feature_list:
        beta.setdefault(feature, {'Unknown':
            {'True': 0, 'False': 0, 'Total': 0}})
    return beta


# Count new entry of feature category information
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


# Randomly select seeds for testing datasets from provided bundle values
def compute_seeds(bundles, sets):
    seeds = []
    for _ in range(sets):
        length = len(bundles)
        seed = [random.choice(range(length)) for i in range(length)]
        seeds.append(seed)
    return seeds


# DEBUG
if __name__ == '__main__':
    print('ERROR: helper.py should not be executed on its own')
