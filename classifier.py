from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import feature_ranking as fr
import os
from tabulate import tabulate


def run_grid_search(data, classes):
    '''
    Returns a dictionary with the results of a grid search over the following parameters:
        solver - [sgd, adam]; momentum / no-momentum
        hidden_layer_sizes - [10, 15, 20]
    The estimator is evaluated using 5 times repeated 2 fold cv.
    Parameters:
        data (pandas dataframe): Dataframe of feature values obtained from loading a csv file.
        classes (numpy array): Array of class id's for each row in a dataframe.
    '''
    param_grid = {
        "solver": ["sgd", "adam"],
        "hidden_layer_sizes": [(10,), (15,), (20,)]
    }
    rkf = RepeatedKFold(n_splits=2, n_repeats=5)

    gs = GridSearchCV(MLPClassifier(), param_grid,
                      scoring='accuracy', n_jobs=-1, cv=rkf)

    gs.fit(data, classes)
    return gs.cv_results_


def pretty_print_cv(values, headers, texfile=None):
    '''
    Dumps a list of tuples into a latex table.
    Parameters:
        results  (list of tuples): List of values for each row in a table.
        headers  (list): List of table header names.
        texfile (str): File path for saving the output. If none, the output is redirected to stdout.
    '''
    table = [list(row) for row in values]
    if texfile:
        latex_out = tabulate(table, tablefmt="latex_raw", headers=headers)
        with open(texfile, "w+") as f:
            f.write(latex_out)
    else:
        print(tabulate(table, tablefmt="grid", headers=headers))
