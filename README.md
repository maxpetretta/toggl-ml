# Toggl-ML
A machine learning project for indentifying modified time entries from a Toggl account, written in Python

## Installation
To compile and run toggl-ml, you will need the following dependencies:
* Python v3.6+
* Pipenv v11.10+

[Pipenv](https://github.com/pypa/pipenv) takes care of all project dependencies automatically.  You can install Pipenv using the following:  

Homebrew: `brew install pipenv`  
Python v3.0+: `pip install pipenv`  

Once Pipenv is installed, navigate to the project directory and run:  

`pipenv install`  

This will install all of the Python dependencies found in `Pipfile`, and create a virtualenv for the project to run in.

## Executing
To run Toggl-ML, make sure you have added your account information to the [keys/](https://github.com/MaxPetretta/toggl-ml/tree/master/keys) directory in the proper format OR by providing your own `data.csv` file in the [data/][https://github.com/MaxPetretta/toggl-ml/tree/master/data] directory.  

All of the learning model options can be configured from `main.py`.  Once you are satisfied with your options, navigate to the top project directory and run:
```
pipenv shell
python src/main.py
```
At this point, the program will collect all data from the provided Toggl account and apply a learning model to the entries.  Output is printed to the console, and several graphs will be shown in separate windows.

To see the graphs again without repeating the export process, simply run:  

`python src/analyse.py`

## Versions
There are two primary versions of this project.  The final version `v1.0` has the model learning on random datasets created by spliting time entries into separate, weekly bundles.  A previous version `v0.3` has the model learning on a shuffled set of all data entries.  These versions can be compared by checking out their respective tags.

---

_Developed by Max Petretta_
