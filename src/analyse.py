# Plot various graphs and distributions of model training data
import os
import csv
import dateutil.parser

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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


# Plot the ending totals of classification values
def plot_confusion(data):
    values = [0, 0, 0, 0]
    labels = ['True Positive', 'True Negative',
              'False Positive', 'False Negative']
    colors = ['green', 'limegreen', 'red', 'darkorange']

    for entry in data:
        modified = entry['modified']
        probability = float(entry['probability'])

        if modified == 'True' and probability >= 0.5:
            values[0] += 1
        elif modified == 'False' and probability < 0.5:
            values[1] += 1
        elif modified == 'False' and probability >= 0.5:
            values[2] += 1
        elif modified == 'True' and probability < 0.5:
            values[3] += 1
    
    # Calculate F1 score, for modified values
    precision = 1 / (1 + (values[2]/values[0]))
    recall = 1 / (1 + (values[3]/values[0]))
    f1_score = 2 / ((1/precision) + (1/recall))
    print('f1_score: ', f1_score)

    # Calculate F2 score, for non-modified values
    precision = 1 / (1 + (values[3]/values[1]))
    recall = 1 / (1 + (values[2]/values[1]))
    f2_score = 2 / ((1/precision) + (1/recall))
    print('f2_score: ', f2_score)

    plt.pie(values, labels=labels, colors=colors, autopct=lambda p:
            round(p/100 * len(data)))
    plt.title('Confusion Matrix Values')


# Plot the progression of the misclassification error rate
def plot_misclassification(data):
    x, y = [], []
    for index, entry in enumerate(data, 1):
        x.append(index)
        y.append(float(entry['error']))
    
    print('error rate: ', y[-1])
           
    plt.scatter(x, y)
    plt.title('Misclassification Rate')
    plt.xlabel('Entry Number')
    plt.ylabel('Error Rate (Percentage)')
    plt.grid(linestyle='--')


# Plot the progression of the cross entropy rate
def plot_entropy(data):
    x, y = [], []
    for index, entry in enumerate(data, 1):
        x.append(index)
        y.append(float(entry['entropy']))
    
    print('entropy rate: ', y[-1])
           
    plt.scatter(x, y)
    plt.title('Entropy Rate')
    plt.xlabel('Entry Number')
    plt.ylabel('Entropy Rate (Percentage)')
    plt.grid(linestyle='--')


# Plot the histogram of entry start/end times
def plot_times(data, target):
    times = []
    for entry in data:
        date = dateutil.parser.parse(entry[target])
        times.append(date.hour)
  
    plt.hist(times)
    plt.title(f"{target.capitalize()} Time Distribution")
    plt.xlabel('Time (By Hour)')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--')
    plt.xticks(range(0, 24, 2))


# Plot the histogram of entry durations
def plot_duration(data):
    durations = []
    for entry in data:
        minutes = (int(entry['duration'])/1000) / 60
        durations.append(round(minutes))
      
    plt.hist(durations)
    plt.title('Duration Distribution')
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--')
    

# Visually examine data after being processed through model
def analyse():
    print('\nANALYSE:')

    # Open processed model data
    with open(os.path.join(data_path, 'model.csv')) as file:
        data = open_csv(file)
    
    # Show confusion matrix values in pie chart
    plt.figure(num=1, figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plot_confusion(data)
    
    # Show misclassification rate in scatter plot
    plt.subplot(1, 3, 2)
    plot_misclassification(data)

    # Show cross entropy rate in scatter plot
    plt.subplot(1, 3, 3)
    plot_entropy(data)

    print('Showing confusion, misclassification, and entropy results')
    plt.show()
    
    # Show distribution of starting times
    plt.figure(num=2, figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plot_times(data, 'start')

    # Show distribution of ending times
    plt.subplot(1, 3, 2)
    plot_times(data, 'end')

    # Show distribution of entry durations
    plt.subplot(1, 3, 3)
    plot_duration(data)

    print('Showing time distribution results')
    plt.show()


# DEBUG
if __name__ == '__main__':
    analyse()
