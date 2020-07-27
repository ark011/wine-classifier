import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Model:

	def __init__(self, model_path: str = None):
		self._model = None
		self._model_path = model_path
		self.load()


	def train(self, X: np.ndarray, y : np.ndarray):
		self._model = LogisticRegression()
		self._model.fit(X,y)
		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self._model.predict(X)

	def save(self):
		if self._model is not None:
			pickle.dump(self._model, open(self._model_path, 'wb'))
		else:
			raise TypeError("The model has not been trained, please include .train() in order to save the model")

	def load(self):
		try:
			self._model = pickle.load(open(self._model_path, 'rb'))
			print("Successfully loaded model...")
		except:
			self._model = None
			print("Error occurred. Model cannot be loaded...")
		print(self._model)
		return self

model_path = Path(__file__).parent / "model.pkl"
num_attributes = pd.read_csv('/api/ml/wine_combined.csv').shape[1]-1
model = Model(model_path)

def get_model():
	return model


if __name__ == "__main__":
	df = pd.read_csv('/api/ml/wine_combined.csv')
	X = df.iloc[:,:-1]
	y = np.array(df['wine_type'])
	print(X.head())
	print(y)
	model.train(X,y)
	model.save()