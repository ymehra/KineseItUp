import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import script as sc
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

user = 'Hans'
input_file = 'complete.csv'

data = pd.DataFrame(sc.get_complete(user,input_file))
data = sc.get_lags(data, 1)
data = data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'start.time', 'index', 'activity'], axis=1)
data_with_predictions = pd.DataFrame()

## define y variable
y_var = 'coding'

data_x = data.drop(['type','coding'], axis = 1)
data_y = data[y_var]

x_vars = data_x.columns

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(max_depth=5,
                              random_state=0)

forest.fit(data_x, data_y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(data_x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print (x_vars[indices[f]])

# Plot the feature importances of the forest
plt.figure(figsize = (8,6))
plt.title("Feature importances")
plt.bar(range(data_x.shape[1]), importances[indices],
       color="hotpink", yerr=std[indices] / 13, align="center")
plt.xticks(range(data_x.shape[1]), x_vars[indices],rotation = 90,fontsize= 8)
plt.xlim([-1, data_x.shape[1]])
plt.gcf().subplots_adjust(bottom=0.20)
plt.savefig('feature_importances.png')