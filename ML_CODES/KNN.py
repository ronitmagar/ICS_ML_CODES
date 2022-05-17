# import packages
import numpy as np
import pandas as pd

# create dataset
x = [[2, 4], [4, 4], [4, 6], [4, 2], [6, 2], [6, 4], [8, 2]]
y = ["Orange", "Blue", "Orange", "Orange", "Blue", "Orange", "Blue"]
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)  #n_neighbors indicates 'K'

# train the algorithm
classifier.fit(x,y)

# predict class for points (6,6)
x_test = np.array([6,6])
y_pred = classifier.predict([x_test])
print(y_pred)