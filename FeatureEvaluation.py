import numpy as np
from sklearn.model_selection import train_test_split

import classification
import feautureSelection
import splitData
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

seed = 57


clf = tree.DecisionTreeClassifier()

for i in range(1, 16):
    output, text = feautureSelection.selectFeature(i, splitData.x)
    x_train, x_test, y_train, y_test = train_test_split(output, splitData.y, random_state=seed, test_size=0.2)
    x_train = np.reshape(x_train, (-1, 1))

    clf.fit(x_train, y_train)
    #
    y_pred = clf.predict(y_test)
    print(f'F1-score of {text}={f1_score(y_test, y_pred)}')
    print(f'Confusion Matrix of {text}= \n{confusion_matrix(y_test, y_pred)}')


