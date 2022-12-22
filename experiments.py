import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.model_selection import cross_val_score
import random
import os
from sklearn import metrics
import feautureSelection
import classification


def normalization(x):
    min_i = np.min(x, axis=1)
    max_i = np.max(x, axis=1)
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = ((x[i][j]-min_i[i])/(max_i[i]-min_i[i]))
    return x


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

# norm_x = Normalizer().fit(x)
# normalized_x = norm_x.transform(x)
# normalized_x = normalization(x)

output = feautureSelection.feature_engineering(x)
x_train, x_test, y_train, y_test = train_test_split(output, y, random_state=seed, test_size=0.2)

f = open("output2.txt", "w")


classification.svm(x_train, y_train, x_test, y_test, f)
classification.random_forest(x_train, y_train, x_test, y_test, f)
classification.knn(x_train, y_train, x_test, y_test, f)

f.close()

fig, ax = plt.subplots(figsize=(6, 6))
clf = classification.classifiers
pred = classification.predicts
# print(len(clf), len(pred))
for i in range(len(clf)):
    text = ''
    if i < 4:
        text = f'svm {i}'
    elif 4 <= i < 103:
        text = f'random forest where depth= {i-3}'
    else:
        text = f'KNN when k = {i - 102 }'
    if 0 <= i < 4 or i == 13 or i == 107:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred[i])
        plt.plot(fpr, tpr, label=text)

plt.legend()
plt.show()
