import numpy as np
import xlsxwriter

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.model_selection import cross_val_score
import random
import os
from sklearn import metrics
from sklearn.preprocessing import normalize

import feautureSelection
import classification



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

normalized_x = normalize(x,axis=0)
# normalized_x = normalization(x)

output = feautureSelection.feature_engineering(x)
x_train, x_test, y_train, y_test = train_test_split(output, y, random_state=seed, test_size=0.2)

f = open("output3.txt", "w")
workbook = xlsxwriter.Workbook('f1.xlsx')
worksheet = workbook.add_worksheet()

classification.svm(x_train, y_train, x_test, y_test, worksheet)
classification.random_forest(x_train, y_train, x_test, y_test, worksheet)
classification.knn(x_train, y_train, x_test, y_test, worksheet)

f.close()
workbook.close()

fig, ax = plt.subplots(figsize=(6, 6))
clf = classification.classifiers
pred = classification.predicts

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred[0])
plt.plot(fpr, tpr, label="SVM, kernel = linear")

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred[16])
plt.plot(fpr, tpr, label="Random Forest, max depth=13")

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred[112])
plt.plot(fpr, tpr, label="KNN, K=6")

plt.legend()
plt.show()
