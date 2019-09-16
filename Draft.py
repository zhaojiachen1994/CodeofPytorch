# -*- coding: utf-8 -*-
"""
File: Draft.py
Project: CodeofPytorch
Author: Jiachen Zhao
Date: 9/16/19
Description:
"""
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1,n_samples=1000)
rng = np.random.RandomState(2)
X += 0*rng.uniform(size=X.shape)
# X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

svm = SVC(gamma=2, C=1)
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('svm accuracy:', score)


if __name__ == "__main__":
    print('--------')