import numpy as np
import pandas as pd

from typing import List
from fastapi import FastAPI, Depends
from pydantic import BaseModel, validator

from .ml.model import Model, get_model, num_attributes

""" Base model class that is used to for prediction. validator included to make sure 
	that the data point contains the right amount of attributes/features in order to predict
"""
class PredictRequest(BaseModel):
	data: List[List[float]]

	@validator("data")
	def check_dimensions(cls, incoming_data):
		for data_point in incoming_data:
			if len(data_point) != num_attributes:
				raise ValueError(f"Each data point must contain {num_attributes} attributes/features")
		return incoming_data


""" Base model class that is used to return the data """
class PredictResponse(BaseModel):
	data: List[str]


app = FastAPI()

""" Display Welcome message on home endpoint"""
@app.get('/')
def read_root():
	return "Welcome!"


""" POST METHOD to pass in the data and predict whether its red or white wine"""
@app.post("/predict", response_model=PredictResponse)
def predict(input: PredictRequest, model: Model = Depends(get_model)):
	X = np.array(input.data)
	y_pred = model.predict(X)
	result = PredictResponse(data=y_pred.tolist())
	return result


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=80)