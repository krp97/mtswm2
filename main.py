
import sys
import os
import feature_ranking as fr


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def main():
    csv_path = os.path.join(get_script_path(), 'data/brzuch_dane.csv')
    x, y = fr.load_data(csv_path)
    ranking = fr.rank_features(x, y, sort=True)
    fr.pretty_print_ranking(ranking)


if __name__ == '__main__':
    main()
