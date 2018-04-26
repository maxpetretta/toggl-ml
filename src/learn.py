# Apply live probability classifiers to entries in passed data set
import os
import csv
import math
import dateutil.parser
import helper as h
from scipy import special


# Global variables
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data/')


# Divide passed data into bundles of chronological data, spliting by days
def split_data(data, days):
    index = 0
    start = 0
    count = days
    bundles = []

    for entry in reversed(data):
        current = dateutil.parser.parse(entry['start']).day

        # Create a new bundle if the index has been incremented
        if index >= len(bundles):
            start = current
            bundles.append([])
        
        # Update start value if the current day value changes
        elif current != start:
            count -= 1
            start = current

        bundles[index].append(entry)

        # Reset the day counter once number of days has passed
        if count == 0:
            index += 1
            count = days
    return bundles


# Determine probabilities for categorical features on time entry
def compute_prob_categorical(entry, previous):

    # Update previous beta features with new entry
    beta = (previous['beta'] if previous is not None else h.setup_beta())
    beta = h.update_beta(entry, beta)
    
    # Count total number of prior manual updates, termed as alpha
    alpha = h.compute_alpha(beta)

    # Compute prior probability of an entry being modified, termed as theta
    theta = alpha[1] / (alpha[0] + alpha[1])
    
    # Find logarithmic probability for the categorical features and save values
    prob_categorical = (math.log(theta / (1-theta))
                        + h.sum_log_ratios(entry, beta))
    entry['beta'] = beta

    return (entry, prob_categorical)


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
        a_0, b_0 = h.update_hyperparameters(x, a_0, b_0, kappa)
    else:
        a_1, b_1 = h.update_hyperparameters(x, a_1, b_1, kappa)
    
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

    # Log of the ratio of the probability of duration
    term_cd = (d_0*c_1 - d_1*c_0) / (d_0*d_1)
    term_log_cd = math.log(c_1/c_0) + math.log(d_0 / d_1)

    # Compute final duration probability and save values
    prob_duration = term_cd * (-x) * term_log_cd
    entry['c_0'], entry['d_0'] = c_0, d_0
    entry['c_1'], entry['d_1'] = c_1, d_1

    return (entry, prob_duration)


# Determine true probability of the entry being modified from it's features
def compute_prob_sigmoid(entry, p1, p2, p3, p4):
    
    # Sum all probability values
    prob_sum = p1 + p2 + p3 + p4
    
    # Convert to sigmoid probability, where 0.5 divides false from true
    probability = 1 / (1 + math.e**(-prob_sum))
    entry['probability'] = probability
    return entry
    

# Find the total number of misclassifications to show the mean error rate and 
def compute_error(entry, previous, errors, count):

    # Retrieve prior entropy rates and count any new errors
    entropy_loss = 0
    prior_entropy_loss = (previous['entropy'] if previous is not None else 0)
    log_ratio = math.log((1 - entry['probability']) / entry['probability'])

    if entry['modified'] == 'False' and entry['probability'] >= 0.5:
        errors += 1     # False positive
        entropy_loss -= log_ratio
    elif entry['modified'] == 'True' and entry['probability'] < 0.5:
        errors += 1     # False negative
        entropy_loss += log_ratio
    
    # Find new mean error rate and entropy loss rate
    ratio = 1 / count
    error_rate = ratio * errors
    entry['error'] = error_rate

    entropy_rate = ((1-ratio) * prior_entropy_loss) + (ratio * entropy_loss)
    entry['entropy'] = entropy_rate

    return (entry, errors)


# Run the probability model for the data contained in the bundle
def run_model(bundles, index, kappa_start, kappa_end):
    errors = 0
    bundle = bundles[index]

    for count, entry in enumerate(bundle, 1):
        entry['bundle'] = index
        previous = (bundle[count - 2] if count > 1 else None)

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
        entry = compute_prob_sigmoid(entry, prob_categorical,
            prob_time_start, prob_time_end, prob_duration)
        
        # Compute misclassification error rate of model
        entry, errors = compute_error(entry, previous, errors, count)
        
    # Output bundle results
    last = bundles[index][-1]
    _, scores = h.compute_scores(bundles[index])
    print(f"Bundle: {last['bundle']}\n",
            f" Misclassification Rate: {last['error']}\n",
            f" Entropy Rate: {last['entropy']}\n",
            f" F1 Score (Modified): {scores[0]}\n",
            f" F2 Score (Not Modified): {scores[1]}\n")


# Find mean error rates and scores for random sets of bundles
def compute_datasets(bundles, seeds):
    datasets = []

    for seed in seeds:
        datasets.append({})
        datasets[-1]['seed'] = seed

        # Sum all values within the set defined by the seed
        error_rate = 0
        entropy_loss = 0
        scores = [0, 0]
        for i in seed:
            error_rate += bundles[i][-1]['error']
            entropy_loss += bundles[i][-1]['entropy']
            _, new_scores = h.compute_scores(bundles[i])
            scores = [scores[i] + new_scores[i] for i in range(len(scores))]

        # Find the mean values by dividing by the seed size
        error_rate /= len(seed)
        entropy_loss /= len(seed)
        scores = [score / len(seed) for score in scores]

        # Save all values onto the dataset dictionary
        datasets[-1]['error'] = error_rate
        datasets[-1]['entropy'] = entropy_loss
        datasets[-1]['f1'] = scores[0]
        datasets[-1]['f2'] = scores[1]
    return datasets


# Examine time entries, building live probability model
def learn(days, sets):
    print('\nLEARN:')

    # Open training data set
    with open(os.path.join(data_path, 'train.csv')) as file:
        data = h.open_csv(file)
    print(f"Training model using data from train.csv",
          f"({len(data)} entries)")
    
    # Split training data into separate bundles, dividing by number of days
    bundles = split_data(data, days)

    # Loop over bundle data for live model learning
    kappa_start = h.compute_kappa(data, 'start')
    kappa_end = h.compute_kappa(data, 'end')
            
    for index, bundle in enumerate(bundles):
        run_model(bundles, index, kappa_start, kappa_end)
    
    # Generate random dataset seeds, using bundles as building blocks
    seeds = h.compute_seeds(bundles, sets)

    # Calculate mean result rates from dataset of seed values
    datasets = compute_datasets(bundles, seeds)
    for count, dataset in enumerate(datasets):
        print(f"Dataset: {count}  -  {dataset['seed']}\n",
              f" Misclassification Rate: {dataset['error']}\n",
              f" Entropy Rate: {dataset['entropy']}\n",
              f" F1 Score (Modified): {dataset['f1']}\n",
              f" F2 Score (Not Modified): {dataset['f2']}\n")
    
    # Save updated training data to new .csv file
    keys = bundles[0][0].keys()
    with open(os.path.join(data_path, 'model.csv'), 'w') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for bundle in bundles:
            writer.writerows(bundle)
    
    # Save dataset results to new .csv file
    keys = datasets[0].keys()
    with open(os.path.join(data_path, 'output.csv'), 'w') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(datasets)
    print('Finished training model...')


# DEBUG
if __name__ == '__main__':
    learn(7, 100)
