from sklearn.ensemble import IsolationForest


X = [[1.1], [0.3], [0.5], [100]]
clf = IsolationForest(random_state=0).fit(X)
print(clf.predict([[0.1], [0], [90]]))
print(clf.decision_function([[0.1], [0], [90]]))