FROM python:3.6-slim


COPY ./api /api/api
COPY ./api/ml/wine_combined.csv /api/ml/wine_combined.csv
COPY ./api/ml/model.pkl /api/ml/model.pkl
COPY requirements.txt /requirements.txt

RUN apt-get update \
    && apt-get install python-dev python-pip -y \
    && pip install -r requirements.txt

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 80

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0", "--port", "80"] 
