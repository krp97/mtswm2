
import warnings
import sys
import os
import feature_ranking as fr
import classifier as clsf


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def basic_rank():
    csv_path = os.path.join(get_script_path(), 'data/brzuch_dane.csv')
    x, y = fr.load_data(csv_path)
    ranking = fr.rank_features(x, y, sort=True)
    fr.pretty_print_ranking(ranking)


def parse_into_table(results):
    '''
    Parses the experiment output into table format.
    Returns a 2 dimensional list of values.

    Parameters:
        results (list(tuple(dict, int))): Output retrieved from running 'run_experiment'.
    '''
    output = []
    for res, feature_count in results:
        for item, param in enumerate(res['params']):
            row = [feature_count]
            layer_size, = param['hidden_layer_sizes'] # --> note: hidden_layer_sizes is a tuple
            row.append(layer_size)
            row.append(
                'Tak') if param['solver'] == 'sgd' else row.append('Nie')
            avg_score = res['mean_test_score'][item] * 100
            row.append("{0:.2f}".format(avg_score))
            output.extend([row])

    return output


def run_experiment(x, y, upper_bound=5):
    '''
    Runs the experiment 'upper_bound' amount of times.
    Each iteration, another feature from the ranking gets added to the grid search process.
    Results of each iteration are dumped into a latex table.

    Parameters:
        x (pandas dataframe): Dataframe obtained from loading a csv file.
        y (numpy array): Array of class id's for each row in a dataframe.
        upper_bound (int): Last amount of features before the experiment exits.
    '''
    results = []
    for i in range(1, upper_bound):
        new_x, new_y = fr.rank_n_transform(x, y, i)
        partial_res = clsf.run_grid_search(new_x, new_y)
        results.append((partial_res, i))

    table_data = parse_into_table(results)
    headers = ['Liczba cech', 'Liczba neuronów', 'Momentum', 'Skuteczność [\%]']
    clsf.pretty_print_cv(table_data, headers, texfile='hello.tex')


def print_confusion_matrix(x, y, feature_count):
    x_new, y_new = fr.rank_n_transform(x, y, 8)
    clsf.get_conf_matrix(x_new, y_new, hidden_layer_sizes=(20,), solver='adam')

def main():
    csv_path = os.path.join(get_script_path(), 'data/brzuch_dane.csv')
    x, y = fr.load_data(csv_path)
    run_experiment(x, y, upper_bound=3)
    print_confusion_matrix(x, y, 8)


if __name__ == '__main__':
    main()
