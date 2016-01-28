from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def classify(features_train, labels_train):   
	clf = GaussianNB()
	clf.fit(features_train,labels_train)
	pred = clf.predict(features_train)
	print accuracy_score(pred, labels_train)
	return  clf.fit(features_train, labels_train)

#printing the accuracy:
# accuracy = no. of points classified correctly / all points (in test set)