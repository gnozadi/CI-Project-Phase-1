import pickle

import collections
import math

import numpy
import numpy as np
from scipy.stats import *


def mean(x):  # 1 - Statistical
    return np.mean(x, axis=1)


def median(x):  # 2 - Statistical
    return np.median(x, axis=1)


def std(x):  # 3 - Statistical
    return np.std(x, axis=1)


def skw(x):  # 4 - Statistical
    return skew(x, axis=1)


def krt(x):  # 5 - Statistical
    return kurtosis(x, axis=1)


def max(x):  # 6 - Statistical
    return np.max(x, axis=1)


def shannon_i(x):  # 7 - Entropy(1)
    shannon_entropy = 0
    counts = collections.Counter([item for item in x])
    n = len(x)
    for a in counts:
        n_i = counts[a]
        p_i = n_i / n
        entropy_i = p_i * (math.log(p_i, 2))
        shannon_entropy += entropy_i
    return shannon_entropy


def shannon(x):
    arr = []
    for i in range(len(x)):
        arr.append(shannon_i(x[i]))
    return np.array(arr)


def log_energy_i(x):  # 8 - Entropy(2)
    n = 4097
    i = 1
    sum = 0
    while i < n:
        s_n = math.pow(x[i], 2)
        if s_n != 0:
            sum += math.log(s_n, 10)
        i += 1
    return sum


def log_energy(x):
    arr = []
    for i in range(len(x)):
        arr.append(log_energy_i(x[i]))
    return np.array(arr)


def renyi_i(x):  # 9 - Entropy(3)
    counts = collections.Counter([item for item in x])
    a = 2
    sum = 0
    for a in counts:
        p_i = counts[a] / len(x)
        sum += p_i
    return 1 - sum


def renyi(x):
    arr = []
    for i in range(len(x)):
        arr.append(renyi_i(x[i]))
    return np.array(arr)


def norm_i(x):  # 10 - Entropy(4)
    h = 1.1
    sum = 0
    for i in range(len(x)):
        sum += math.pow(abs(x[i]), h)
    return sum


def norm(x):
    arr = []
    for i in range(len(x)):
        arr.append(norm_i(x[i]))
    return np.array(arr)


def ptp(x):
    min_i = np.min(x, axis=1)
    max_i = np.max(x, axis=1)
    return max_i - min_i


def zc_i(x):
    count = 0
    for i in range(len(x)):
        if x[i] == 0:
            count += 1
    return count


def zc(x):
    arr = []
    for i in range(len(x)):
        arr.append(zc_i(x[i]))
    return np.array(arr)


def fourier_transform(x):
    # arr = []
    # for i in range(len(x)):
    # arr.append()
    return np.fft.fft(x)
    # return np.array(arr)

def feature_engineering(x):
    out = np.vstack((median(x), mean(x), std(x), skw(x), krt(x),max(x),shannon(x),log_energy(x),renyi(x), norm(x), ptp(x), zc(x)))
    return out.transpose()

x = pickle.load(open('x.pkl', 'rb'))
# print(fft_i(x))
# print(fourier_transform(x))
print(feature_engineering(x).shape)