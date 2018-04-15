# Plot graphs and distributions of model training data
import os
import csv
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


# Plot the ending totals of false positive/negative values
def plot_confusion(data):
    values = [0, 0, 0, 0]
    labels = ['True Positive', 'True Negative',
              'False Positive', 'False Negative']

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

    plt.pie(values, labels=labels, autopct=lambda p: round(p/100 * len(data), 0))
    plt.show()


# Plot the progression of misclassification 
def plot_misclassification(data):
    x = []
    y = []
    for index, entry in enumerate(data, 1):
        x.append(index)
        y.append(float(entry['error']))
    plt.scatter(x, y)
    plt.show()


# Visually examine data after being processed through the model
def analyse():
    print('\nANALYSE:')

    # Open processed model data
    with open(os.path.join(data_path, 'model.csv')) as file:
        data = open_csv(file)
    
    # Show confusion matrix values in pie chart
    plot_confusion(data)
    
    # Show misclassification rate in scatter plot
    plot_misclassification(data)




# DEBUG
if __name__ == '__main__':
    analyse()
