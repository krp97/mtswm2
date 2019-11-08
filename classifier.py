from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import feature_ranking as fr

data, classes = fr.load_data('./data/brzuch_dane.csv')

param_grid = {
    "solver": ["sgd", "adam"],
    "hidden_layer_sizes": [(10,), (15,), (20,)]
}
rkf = RepeatedKFold(n_splits=2, n_repeats=5)

gs = GridSearchCV(MLPClassifier(), param_grid,
                  scoring='accuracy', n_jobs=-1, cv=rkf)

gs.fit(data, classes)
print(gs.cv_results_)
