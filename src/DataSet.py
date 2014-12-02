
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing


class DataSet:
	def __init__(self, X, y, name, X_test=None, y_test=None, norm=False):
		if norm:
			self.X = normalize(X)
		else:
			self.X = X
		self.y = y
		self.name = name

		# preprocessing
		X = preprocessing.scale(X)

		if not X_test and not y_test:
			self.X, _X_test, self.y, _y_test = train_test_split(self.X, self.y, test_size=0.3)
			
			# for now
			self.Xtest = _X_test
			self.ytest = _y_test

	def get_cv_performance(self, clf):
		X = self.X
		y = self.y
		scores = cross_validation.cross_val_score(clf, X, y, cv=cross_validation.KFold(n=X.shape[0], n_folds=10))#cv=10)
		print("%s error rate: %0.2f (+/- %0.2f)" % (self.name, 1.-scores.mean(), scores.std() * 2))
