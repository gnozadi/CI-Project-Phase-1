import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import random
import os

import feautureSelection

seed = 57

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

x = pickle.load(open('x.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))


x_normal = np.concatenate((x[:300], x[400:]), axis=0)
x_seizure = x[300:400]
# print(x_normal.shape)
# print(x_seizure.shape)
sampling_freq = 173.6  # based on info from website

# keeping signals between 0.5 to 40 hz
b, a = butter(3, [0.5, 40], btype='bandpass', fs=sampling_freq)

x_normal_filtered = np.array([lfilter(b, a, x_normal[ind, :]) for ind in range(x_normal.shape[0])])
x_seizure_filtered = np.array([lfilter(b, a, x_seizure[ind, :]) for ind in range(x_seizure.shape[0])])


x_normal = x_normal_filtered
x_seizure = x_seizure_filtered

x = np.concatenate((x_normal, x_seizure))
y = np.concatenate((np.zeros((400, 1)), np.ones((100, 1))))

output = feautureSelection.feature_engineering(x)
x_train, x_test, y_train, y_test = train_test_split(output, y, random_state=seed, test_size=0.2)


# clf1 = SVC(kernel='linear')
# clf2 = RandomForestClassifier(max_depth=2)
# clf3 = KNeighborsClassifier()
# clf4 = KNeighborsClassifier(n_neighbors=9)
#
#
# # clf1.fit(x_train, y_train)
# # clf2.fit(x_train, y_train)
# clf3.fit(x_train,y_train)
# clf4.fit(x_train,y_train)
#
# # y_pred = clf1.predict(x_test)
# # y_pred2 = clf2.predict(x_test)
# y_pred3 = clf3.predict(x_test)
# y_pred4 = clf4.predict(x_test)
#
# # print(accuracy_score(y_test, y_pred))
# # print(accuracy_score(y_test, y_pred2))
# print(accuracy_score(y_test, y_pred3))
# print(accuracy_score(y_test, y_pred4))

print("SVM")
clf_svm = SVC(kernel='linear')
clf_svm.fit(x_train, y_train)
pred = clf_svm.predict(x_test)
print(accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred))
clf_svm = SVC(kernel='poly')
clf_svm.fit(x_train, y_train)
pred = clf_svm.predict(x_test)
print(accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred))
clf_svm = SVC(kernel='rbf')
clf_svm.fit(x_train, y_train)
pred = clf_svm.predict(x_test)
print(accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred))
clf_svm = SVC(kernel='sigmoid')
clf_svm.fit(x_train, y_train)
pred = clf_svm.predict(x_test)
print(accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred))

print("Random Forest")
clf_rf = RandomForestClassifier(criterion='gini')
clf_rf.fit(x_train, y_train)
pred = clf_rf.predict(x_test)
print(accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred))
clf_rf = RandomForestClassifier(criterion='entropy')
clf_rf.fit(x_train, y_train)
pred = clf_rf.predict(x_test)
print(accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred))
clf_rf = RandomForestClassifier(criterion='log_loss')
clf_rf.fit(x_train, y_train)
pred = clf_rf.predict(x_test)
print(accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred))

print("KNN")
for i in range(1,100):
    clf_KNN = KNeighborsClassifier(n_neighbors=i)
    clf_KNN.fit(x_train, y_train)
    pred = clf_KNN.predict(x_test)
    print(i, accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred))




