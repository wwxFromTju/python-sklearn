#!/usr/bin/env python
# encoding=utf-8

import csv
from collections import defaultdict

import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

estimator = KNeighborsClassifier()

X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

with open('data/Ionosphere.data', 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        X[i] = data
        y[i] = row[-1] == 'g'


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
print("There are {} samples in the training dataset".format(X_train.shape[0]))
print("There are {} samples in the testing dataset".format(X_test.shape[0]))
print("Each sample has {} features\n".format(X_train.shape[1]))

estimator.fit(X_train, y_train)
print(estimator, '\n')

y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print('The accuracy is {0:.1f}%'.format(accuracy))

scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print('\nThe average accuracy is {0:.1f}%'.format(average_accuracy))

avg_scores = []
all_scores = []
parameter_value = list(range(1, 21))
for n_neighbors in parameter_value:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

plt.figure(figsize=(10, 6), dpi=128)
plt.plot(parameter_value, avg_scores, '-o', linewidth=1, markersize=10)
plt.show()

plt.figure(figsize=(10, 6), dpi=128)
for parameter, scores in zip(parameter_value, all_scores):
    n_scores = len(scores)
    plt.plot([parameter] * n_scores, scores, '-o')
plt.show()

plt.figure(figsize=(10, 6), dpi=128)
plt.plot(parameter_value, all_scores, 'bx')
plt.show()

all_scores = defaultdict(list)
parameter_values = list(range(1, 21))  # Including 20
for n_neighbors in parameter_values:
    for i in range(100):
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(estimator, X, y, scoring='accuracy', cv=10)
        all_scores[n_neighbors].append(scores)
for parameter in parameter_values:
    scores = all_scores[parameter]
    n_scores = len(scores)
    plt.plot([parameter] * n_scores, scores, '-o')
plt.show()

plt.plot(parameter_values, avg_scores, '-o')
plt.show()
