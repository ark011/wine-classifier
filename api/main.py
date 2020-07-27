import numpy as np
import pandas as pd

from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel, ValidationError, validator
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from .ml.model import Model, get_model, num_attributes

class PredictRequest(BaseModel):
	data: List[List[float]]

	@validator("data")
	def check_dimensionality(cls, v):
		for data_point in v:
			if len(data_point) != num_attributes:
				raise ValueError(f"Each data point must contain {num_attributes} attributes/features")
		return v

class PredictResponse(BaseModel):
	data: List[str]


app = FastAPI()

@app.get('/')
def read_root():
	return "Welcome!"


@app.post("/predict", response_model=PredictResponse)
def predict(input: PredictRequest, model: Model = Depends(get_model)):
	X = np.array(input.data)
	y_pred = model.predict(X)
	result = PredictResponse(data=y_pred.tolist())
	return result


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=80)