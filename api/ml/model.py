import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression

class Model:

	""" initialize model and its fields """
	def __init__(self, model_path: str = None):
		self._model = None
		self._model_path = model_path
		self.load()


	""" train function, takes in two numpy arrays
		X - attributes
		y - label
		Logistic Regression model is loaded and model is fitted with X and y
	"""
	def train(self, X: np.ndarray, y : np.ndarray):
		self._model = LogisticRegression()
		self._model.fit(X,y)
		return self

	""" predict function, takes in X numpy array, returns prediction """
	def predict(self, X: np.ndarray) -> np.ndarray:
		return self._model.predict(X)

	""" save function, to save the model if model does not exist, raises an error """
	def save(self):
		if self._model is not None:
			pickle.dump(self._model, open(self._model_path, 'wb'))
		else:
			raise TypeError("The model has not been trained, please include .train() in order to save the model")

	""" load function, loads the model that has been trained into _model field """
	def load(self):
		try:
			self._model = pickle.load(open(self._model_path, 'rb'))
			print("Successfully loaded model...")
		except:
			self._model = None
			print("Error occurred. Model cannot be loaded...")
		print(self._model)
		return self

# get models' path
model_path = Path(__file__).parent / "model.pkl"

# number of attributes, used in main.py to validate the data that is passed in 
# has the correct number of attributes/features
num_attributes = pd.read_csv('/api/ml/wine_combined.csv').shape[1]-1

# call to instantiate Model class and creates instance
model = Model(model_path)

def get_model():
	return model

if __name__ == "__main__":
	df = pd.read_csv('/api/ml/wine_combined.csv') 	# read csv file
	X = df.iloc[:,:-1] 								# take X attributes/features
	y = np.array(df['wine_type'])					# take y label
	model.train(X,y)								# train model
	model.save()									# save model