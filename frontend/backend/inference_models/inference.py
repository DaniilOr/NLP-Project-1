import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch
import torch.nn as nn
import numpy as np
import os
import transformers
import requests

FACT_API_URL = "https://api-inference.huggingface.co/models/stealthpy/sb-temfac"
fact_headers = {"Authorization": "Bearer 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"}

BIAS_API_URL = "https://api-inference.huggingface.co/models/theArif/mbzuai-political-bias-bert"
bias_headers = {"Authorization": "Bearer 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"}

def query_bias(payload):
	response = requests.post(BIAS_API_URL, headers=bias_headers, json=payload)
	return response.json()

def query_fact(payload):
	response = requests.post(FACT_API_URL, headers=fact_headers, json=payload)
	return response.json()

class ModelInference:
    def __init__(self, inference_type = "factuality") -> None:
        self.inference_type = inference_type

    def predict(self, batch: List[str]) -> np.array:
        res = []
        if self.inference_type == "factuality":
            output = query_fact({
                "inputs": batch,
                "parameters": {
                    'padding':True,
                    'truncation':True,
                    'max_length': 512
                }
            })
            print("Output:", output)
            if 'error' in output:
                time.sleep(30)
                return self.predict(batch)
            for each in output:
                res.append([each[2]['score'], each[1]['score'], each[0]['score']])
        elif self.inference_type == "bias":
            output = query_bias({
                "inputs": batch,
                "parameters": {
                    'padding':True,
                    'truncation':True,
                    'max_length': 512
                }
            })
            print("Output:", output)
            if 'error' in output:
                time.sleep(30)
                return self.predict(batch)
            for each in output:
                res.append([each[0]['score'], each[1]['score'], each[2]['score']])
        res = np.array(res)
        return res

# biasmodel = ModelInference(inference_type="bias")
# biasmodel.predict(["This is a test sentence"])
