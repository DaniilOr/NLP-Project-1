from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import insert
from sqlalchemy.orm import Session
from sqlalchemy import and_
import sys
from os import path
import os
ROOT_DIR = path.dirname(path.abspath(__file__))

sys.path.append(ROOT_DIR)

from database import SessionLocal
from typing import List, Any
#import uvicorn
from scrape.scraping import extract_website
from db_models import Article, Results
from schemas import Article as ART, Results as RES
# from fastapi_cache import FastAPICache
# from fastapi_cache.backends.inmemory import InMemoryBackend
# from fastapi_cache.decorator import cache
from time import time
from more_itertools import chunked
from datetime import timedelta, datetime
#import models.dummy_model_util as dmu
#from memcache import async_memcache as aeromemcached

from inference_models.inference import ModelInference

factmodel = ModelInference(inference_type="factuality")
biasmodel = ModelInference(inference_type="bias")
# app = FastAPI()

from multiprocessing import Pool
from nela_features.nela_features import NELAFeatureExtractor
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nela = NELAFeatureExtractor()


# def request_key_builder(
#     func,
#     namespace: str = "",
#     *,
#     request = None,
#     response = None,
#     #*args,
#     **kwargs,
# ):
#     res = ":".join([
#         namespace,
#         request.method.lower(),
#         request.url.path,
#         repr(sorted(request.query_params.items()))
#     ])
#     return res


def nela_process(text):
    comlexity_vector, complexity_names = nela.extract_complexity(text)
    moral_vector, moral_names = nela.extract_moral(text)
    return {key: val for val, key in zip(comlexity_vector + moral_vector,
                                 complexity_names + moral_names)}



# # Dependency
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


#@app.get("/parse", response_model=List[RES])
#@cache(60, key_builder=request_key_builder)
def parse(url: str, is_forced:bool):
    db = SessionLocal()
    ## Check if it was already analyzed
    result = db.query(Results).join(Article).filter(Article.base_url == url).all()
    result = [r for r in result if (datetime.now() - r.date_added).days <= 7]
    if len(result) != 0 and not is_forced:
        return result
    # If no results
    try:
        result = extract_website(url)
    except Exception as e:
        print(e)
        return {}
    base_url, links = list(result.items())[0]
    articles = [
        Article(base_url=base_url,
                url=links[link]['processed_data'].get('source', ''),
                raw_txt=links[link].get('raw_html', ''),
                txt=links[link]['processed_data'].get('raw_text', ''),
                authors=links[link]['processed_data'].get('author', ''),
                date_created=links[link]['processed_data'].get('date', '')
                )
        for link in links
    ]

    print("Dumping Results")
    results = []
    txts = []
    cur = time()
    for a in articles:
        txts.append(a.txt)
    preds_factuality = []
    preds_bias = []    
    for chunk in chunked(txts[:20], 64):
        biasresults = biasmodel.predict(chunk)
        factresults = factmodel.predict(chunk)
        preds_bias.extend(biasresults)
        preds_factuality.extend(factresults)
    # db.add_all(articles)
    # db.flush()

    pool = Pool(4)
    nela_preds = pool.map(nela_process, txts[:20])
    # print(nela_preds)
    for factresults, biasresults, a, nel in zip(preds_factuality, preds_bias, articles, nela_preds):
        r = Results(
            factuality_results={"Factuality": {"0": "Less Factual", "1": "Mixed Factuality", "2": "Highly Factual"},
             "Scores": {"0": factresults[0], "1": factresults[1], "2": factresults[2]}},
            bias_results={"Bias": {"0": "Left", "1": "Center", "2": "Right"},
             "Scores": {"0": biasresults[0], "1": biasresults[1], "2": biasresults[2]}},
            url_id=a.id,
            nela=nela_preds,
        )
        results.append(r)
        # db.add(r)

    end = time()

    #db.commit()

    print("Time to run: ", end - cur)
    results = [i.__dict__ for i in results]
    print(results)
    return results


# @app.get("/db")
# @cache(expire=60)
# def parse(db: Session = Depends(get_db)):
#     return {"data": db.query(Article).all()}


# @app.get("/urls", response_model=List[str])
# @cache(expire=60,)
def urls():
    print('Getting urls')
    db = SessionLocal()
    return set([a.base_url for a in db.query(Article).all() if (datetime.now() - a.date_added).days <= 7])

# @app.get("/mapped", response_model=List[Any])
# async def parse(db: Session = Depends(get_db)):
#     m = db.query(Results).join(Article).all()
#     return set([str(a.__dict__ )for a in m])

# @app.on_event("startup")
# async def startup():
#     FastAPICache.init(InMemoryBackend(), prefix="fastapi-inmemorycache")
#     print("Started")
#
#
# if __name__ == '__main__':
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
