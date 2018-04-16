# Main script file for running toggl-ml
from export import export
from preprocess import preprocess
from bayes import bayes
from analyse import analyse


# Instructions: Run all script files from the top level directory, calling...
#   $ python src/<SCRIPT>.py


# Module variable definitions:
#   since:          Starting date for exporting time entries
#   until:          Ending date for exporting time entries
#   size_train:     Percentage size of the training data set (0.0 - 1.0)
#   size_test:      Percentage size of the testing data set (0.0 - 1.0)
#   size_validate:  Percentage size of the validation data set (0.0 - 1.0)
#   skip:           Number of lines to skip before printing another outcome

#   NOTE: The three size variables must sum to 1.0, else an error is thrown


# Run Naive Bayes classifier on time entry data from Toggl account
def main(since, until, size_train, size_test, size_validate, skip):
    print('\nMAIN:')

    # Export all data from Toggl account
    export(since, until)

    # Partition data into three separate sets, and find all features
    preprocess(size_train, size_test, size_validate)
    
    # Run Bayesian classification on the test data set, printing outcomes
    bayes(skip)

    # Show visual results of Bayes model of training data
    analyse()

    print('\nFINISHED MAIN\n')


# DEBUG
if __name__ == '__main__':
    main('2018-01-01', '2018-12-31', 0.5, 0.2, 0.3, 20)
