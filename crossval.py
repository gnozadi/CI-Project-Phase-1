import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import random
import os
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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


score_svm = cross_val_score(SVC(kernel='linear'), x, y, cv=5, scoring='accuracy')
score_rf = cross_val_score(RandomForestClassifier(max_depth=14), x, y, scoring='accuracy')
score_knn = cross_val_score(KNeighborsClassifier(n_neighbors=5), x, y, scoring='accuracy')

print(score_svm, score_rf, score_knn)

score_svm = cross_val_score(SVC(kernel='linear'), x, y, cv=5)
score_rf = cross_val_score(RandomForestClassifier(max_depth=14), x, y)
score_knn = cross_val_score(KNeighborsClassifier(n_neighbors=5), x, y)

print(score_svm, score_rf, score_knn)