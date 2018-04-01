# Preprocess time entry data into random training, testing, and validation sets
import os
import csv
import random


# Global variables
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data/')


# Check if passed data set split percentages add to 1.0 (100%)
def verify(size_train, size_test, size_validate):
    if size_train + size_test + size_validate == 1:
        return True
    else:
        return False


# Find all unique values for each category field
def categorize(header, file):
    reader = csv.DictReader(file)
    categories = {}
    
    for row in reader:
        for key, value in row.items():
            if key not in ['project', 'description', 'tags']:
                break
            else:
                categories.setdefault(key, set(['Unknown']))
                categories[key].add(value)

    return categories


# Divide data into three sets based on passed split percentages
def split(data, size_train, size_test, size_validate):
    length = len(data)
    split_train = round(length * size_train)
    split_test = round(length * size_test)

    print(f'Spliting data into: Train ({size_train}), Test ({size_test}), &',
          f'Validate ({size_validate})')
    train = data[:split_train]
    test = data[split_train:split_train + split_test]
    validate = data[split_train + split_test:]

    print('Successfully split data')
    return [train, test, validate]


# Create .csv file from set of entries
def create_file(header, data, file):
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)


# Prepare data for learning model
def preprocess(size_train, size_test, size_validate):

    # Verify set sizes add to 1
    if not verify(size_train, size_test, size_validate):
        print('ERROR: Data set sizes do not add to 1.0')
        return

    # Open .csv file and prepare to partition
    with open(os.path.join(data_path, 'data.csv')) as file:
        reader = csv.reader(file)
        header = next(reader)
        data = list(reader)
    
        # Save lists of category values
        file.seek(0)
        categories = categorize(header, file)

    # Randomize time entries and split into three separate files
    random.shuffle(data)
    train, test, validate = split(data, size_train, size_test, size_validate)

    with open(os.path.join(data_path, 'train.csv'), 'w') as train_file, \
         open(os.path.join(data_path, 'test.csv'), 'w') as test_file, \
         open(os.path.join(data_path, 'validate.csv'), 'w') as validate_file:
        create_file(header, train, train_file)
        create_file(header, test, test_file)
        create_file(header, validate, validate_file)

    print(f"Categories: Projects ({len(categories['project'])}), ",
          f"Descriptions ({len(categories['description'])}), ",
          f"Tags ({len(categories['tags'])})")
    return categories


# DEBUG
if __name__ == '__main__':
    preprocess(0.4, 0.3, 0.3)
