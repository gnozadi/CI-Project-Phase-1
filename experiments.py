import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.preprocessing import Normalizer
import random
import os
from sklearn import metrics
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

normalized_x = Normalizer().fit(x)
output = feautureSelection.feature_engineering(normalized_x)
x_train, x_test, y_train, y_test = train_test_split(output, y, random_state=seed, test_size=0.2)

f = open("normal_output.txt", "w")

classification.svm(x_train, y_train, x_test, y_test, f)
classification.random_forest(x_train, y_train, x_test, y_test, f)
classification.knn(x_train, y_train, x_test, y_test, f)

f.close()

fig, ax = plt.subplots(figsize=(6, 6))
clf = classification.classifiers
pred = classification.predicts
for i in range(len(clf)):
    text = ''
    if i < 4:
        text = f'svm {i}'
    elif 4 <= i < 7:
        text = f'random forest {i}'
    else:
        text = f'KNN when k = {i - 6}'
    if i <= 12:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred[i])
        plt.plot(fpr, tpr, label=text)

plt.legend()
plt.show()
