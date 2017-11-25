import numpy as np

def rmse(a, b):
    sum = 0
    for i in range(len(a)):
        sum = sum + (a[i] - b[i]) ** 2
    return np.sqrt(sum / len(a))

def minMaxNormalizer(data):
    numerator = data - np.min(data)
    denominator = np.max(data) - np.min(data)
    return numerator / (denominator + 1e-7)

def minMaxDeNormalizer(data, originalData):
    shift = np.min(originalData)
    multiplier = np.max(originalData) - np.min(originalData)
    return (data + shift) * multiplier
