# wine-classifier
Machine learning wine classifier model built using FastAPI and Docker

## Overview:
An machine learning model used to predict/classify wines into two different types - white and red using 12 attributes. (fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol) This app was built using Python, FastAPI, and Docker. 

## Install:
```
git clone https://github.com/ark011/wine-classifier.git
```

## Usage:
```
cd wine-classifier
docker build -t wine-classifier .
docker run -p 80:80 wine-classifier
```
## Test:
```
Open POSTMAN, set request to POST and enter 12 float/int values for attributes (fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality	wine_type) into "Body" as JSON object
```
<a href="https://ibb.co/BqDPkzx"><img src="https://i.ibb.co/cD5vqgH/Screen-Shot-2020-07-27-at-3-43-48-PM.png" alt="Screen-Shot-2020-07-27-at-3-43-48-PM" border="0"></a>

## Deployment:
You may deploy this application locally or on a server for example on an AWS EC2 instance.

## Built With:
* Python <a> https://www.python.org/ </a>
* FastAPI <a> https://fastapi.tiangolo.com/ </a>
* Docker <a> https://docs.docker.com/ </a>
