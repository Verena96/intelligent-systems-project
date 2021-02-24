from api import app
from os.path import dirname, realpath
from flask import json
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import json
import pandas as pd

TEST_PRODUCTS_PATH=os.getenv('TEST_PRODUCTS_PATH')

df = pd.read_csv(TEST_PRODUCTS_PATH)
df=df.drop(['product_id','seller_id','creation_date','order_counts'],axis=1)
df=df.dropna()
preprocess = ColumnTransformer(
    [('query_countvec', CountVectorizer(), 'query'),
     ('title_countvec', CountVectorizer(), 'title'),
     ('concatenated_tags_tfidf', TfidfVectorizer(ngram_range=(1,3)), 'concatenated_tags')],
    remainder='passthrough')
df = preprocess.fit_transform(df)

TEST_PRODUCTS_PATH = df.to_json(orient = 'table')

path = dirname(realpath(__file__)) + "\\" + "prod.json"

def carregar_json(camino):
    with open(camino, 'r') as f:
        return json.load(f)

def test_api_works_basic():
    body = carregar_json(path)
    response = app.test_client().post("/v1/categorize", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 200

def test_api_works_products():
    body = carregar_json(TEST_PRODUCTS_PATH)
    response = app.test_client().post("/v1/categorize", data=json.dumps(body), content_type="application/json")
    assert response.status_code == 200

def test_api_bad_request():
    response = app.test_client().post("/v1/categorize", data=json.dumps({}), content_type="application/json")
    assert response.status_code == 400