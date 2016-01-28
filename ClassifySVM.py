# Support Vector Machines

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def classify(features_train, labels_train): 
	clf = SVC(C=1000, kernel='poly', gamma='auto')
	clf.fit(features_train,labels_train)
	pred = clf.predict(features_train)
	print accuracy_score(pred, labels_train)
	return  clf.fit(features_train, labels_train)




#priting the accuracy:
# accuracy = no. of points classified correctly / all points (in test set)