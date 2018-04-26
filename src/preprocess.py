# Split time entry data into random training, testing, and validation sets
import os
import csv
import random


# Global variables
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data/')


# Check if passed data set split percentages add to 1.0 (100%)
def verify(size_train, size_test, size_validate):
    if size_train + size_test + size_validate == 1.0:
        return True
    else:
        return False


# Divide data into three sets based on passed split percentages
def split(data, size_train, size_test, size_validate):
    length = len(data)
    split_train = round(length * size_train)
    split_test = round(length * size_test)

    print(f"Spliting data into: Train ({size_train}), Test ({size_test}), &",
          f"Validate ({size_validate})")
    train = data[:split_train]
    test = data[split_train:split_train + split_test]
    validate = data[split_train + split_test:]
    return (train, test, validate)


# Create .csv file from set of entries
def create_file(header, data, file):
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)


# Prepare data for learning model
def preprocess(size_train, size_test, size_validate):
    print('\nPREPROCESS:')

    # Verify set sizes add to 1
    if not verify(size_train, size_test, size_validate):
        print('ERROR: Data set sizes do not add to 1.0')
        exit()

    # Open .csv file and prepare to partition
    with open(os.path.join(data_path, 'data.csv')) as file:
        reader = csv.reader(file)
        header = next(reader)
        data = list(reader)

    # Split time entries into three separate files
    train, test, validate = split(data, size_train, size_test, size_validate)

    with open(os.path.join(data_path, 'train.csv'), 'w') as train_file, \
         open(os.path.join(data_path, 'test.csv'), 'w') as test_file, \
         open(os.path.join(data_path, 'validate.csv'), 'w') as validate_file:
        create_file(header, train, train_file)
        create_file(header, test, test_file)
        create_file(header, validate, validate_file)


# DEBUG
if __name__ == '__main__':
    preprocess(0.6, 0.2, 0.2)
