import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from scipy.signal import butter, lfilter
from sklearn.model_selection import cross_val_score, KFold
import random
import os
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


# score_svm = cross_val_score(SVC(kernel='linear'), x, y, cv=5)
# score_rf = cross_val_score(RandomForestClassifier(max_depth=14), x, y, cv=5)
# score_knn = cross_val_score(KNeighborsClassifier(n_neighbors=6), x, y, cv=5)
#
# f = open("crossval.txt", "w")
# f.write("SVM\n")
# f.write(f'{score_svm}\n')
# f.write("Random Forest\n")
# f.write(f'{score_rf}\n')
# f.write("KNN\n")
# f.write(f'{score_knn}\n')
# f.close()

kf = KFold(n_splits=5, shuffle=True, random_state=1)
kf.split(x, y)

for train_index, test_index in kf.split(x, y):
    print(f'Train set: {len(train_index)}, Test set:{len(test_index)}')

score_svm = cross_val_score(SVC(kernel='linear'), x, y, cv=kf, scoring='recall')
score_rf = cross_val_score(RandomForestClassifier(max_depth=13), x, y, cv=kf, scoring='recall')
score_knn = cross_val_score(KNeighborsClassifier(n_neighbors=6), x, y, cv=kf, scoring='recall')

f = open("cross_val.txt", "w")
f.write("SVM\n")
f.write(f'{np.mean(score_svm)}\n')
f.write("Random Forest\n")
f.write(f'{np.mean(score_rf)}\n')
f.write("KNN\n")
f.write(f'{np.mean(score_knn)}\n')
f.close()

