import numpy as np
from collections import Counter

# Function for determing the euclidean distance between two input rows of the testing and training datasets.

def ed(row1, row2):

	d = np.sqrt(np.sum((row1-row2)**2))

	return d

class KNN:

	# This alg will check for the closest 4 neighbors of the input data value by default (if k value is not provided by user).

	def __init__(self, k=4):

		self.k = k

	# Function to train the model

	def fit(self, X, y):

		self.X_train = X

		self.y_train = y


	# Function that determines the label of the given url inputs based on the url inputs of the training dataset.

	def determine(self, X):

		labels = []

		for x in X:

			result = self._determine(x)

			labels.append(result)

		return labels


	# Helper function for determining the label for each input url of the testing dataset (whether malicious or benign).

	def _determine(self, x):

		# Determine the distance between this input url to the other input urls of the training dataset.

		d_from_x = []

		for train_input_val in self.X_train:
			
			result = ed(x, train_input_val)

			d_from_x.append(result)

		# Determine k nearest neighbors and their labels

		knn = np.argsort(d_from_x)[:self.k]

		labels_found = [self.y_train[index] for index in knn]

		# Via classification, determine the majority label amongst the k nearest neighbors and label x with that label.

		counts_of_indexes = Counter(labels_found)
		
		mode = counts_of_indexes.most_common()

		return mode[0][0]

		
		

		
		

		