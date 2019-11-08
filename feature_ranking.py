import numpy as np
from data import names
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd


def load_data(csv_path):
    '''
    Returns a tuple (data, classes) where:
        - data - Pandas dataframe without the last(class) column.
        - classes - Numpy array filled with class id's.

    Parameters:
        csv_path (str): Path to the csv file, relative to the project's root directory.
    '''
    data = pd.read_csv(csv_path, sep=',')
    data.columns = names.STP_FEATURES
    classes = data[data.columns[-1]]  # Move classId's to separate arr
    data.drop(data.columns[-1], axis=1, inplace=True)
    return (data, classes)


def rank_n_transform(data, classes, feat_count):
    '''
    Returns a tuple with the transformed feature vector and input class_id vector.

    Parameters:
        data (pandas dataframe): Dataframe obtained from loading a csv file.
        classes (numpy array): Array of class id's for each row in a dataframe.
        feat_count (int): Determines how many features will be selected in KBest transformation.
    '''
    kb_selector = SelectKBest(score_func=chi2, k=feat_count)
    data_new = kb_selector.fit_transform(data, classes)
    return (data_new, classes)


def rank_features(data, classes, sort=True):
    '''
    Returns a dictionary or a list of tuples sorted by value, 
    where the first element is the feature_id and the second
    is it's ranking score. The k parameter specifies how many features should be returned.

    Parameters:
        data (pandas dataframe): Dataframe obtained from loading a csv file.
        classes (numpy array): Array of class id's for each row in a dataframe.
        feat_count (int): Feature count in the destination dictionary.
    '''
    kb_selector = SelectKBest(score_func=chi2, k=data[0].shape)
    kb_selector.fit(data, classes)
    results = kb_selector.scores_
    output = dict(zip(data.columns, results))
    return sorted(output.items(), key=lambda x: x[1], reverse=True) if sort else output


def pretty_print_ranking(ranking):
    for name, val in ranking:
        rounded_val = "{0: .2f}".format(val)
        print(f"{name} : {rounded_val}")
