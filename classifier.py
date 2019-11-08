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


def get_avg_test_score(grid_out, dec=2):
    '''
    Returns the average of 'mean_test_score' from a grid_search run.

    Parameters:
        grid_out (dict): Output from a grid search run.
        dec (int): Specifiy the decimal place alignment of the output value.
    '''
    value = grid_out['mean_test_score']
    percentage = (sum(value)/len(value))*100
    return float("{0:.2f}".format(percentage))


def dump_to_latex(values, headers, filename=None):
    '''
    Dumps a list of tuples into a latex table.
    Parameters:
        results  (list of tuples): List of values for each row in a table.
        headers  (list): List of table header names.
        filename (str): File path for saving the output. If none, the output is redirected to stdout.
    '''
    table = [list(row) for row in values]
    latex_out = tabulate(table, tablefmt="latex_raw", headers=headers)
    if filename:
        with open(filename, "w+") as f:
            f.write(latex_out)
    else:
        print(latex_out)

# Example usage:
# values = [(10, 'abc', 102.12), (20, 'xd', 102.12)]
# dump_to_latex(values, ['xd', 'xd', 'xd'], 'output.txt')
