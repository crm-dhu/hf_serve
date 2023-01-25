from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline

# load the model bert-base-uncased
unmasker = pipeline('fill-mask', model='bert-base-uncased')

app = FastAPI()

class ModelInput(BaseModel):
    input: str

class ModelPrediction(BaseModel):
    score: float
    token: int
    token_str: str
    sequence: str

class ModelOutput(BaseModel):
    input: str
    output: List[ModelPrediction]

@app.get("/")
async def hello():
    return "use endpoint /unmask and provide a masked_str with [MASK] filling the token being masked."

@app.post("/unmask")
async def unmask(payload: ModelInput) -> ModelOutput:
    masked_str = payload.input
    unmasked_res = unmasked(masked_str)
    return ModelOutput(input=masked_str, output=unmasked_res) 

def unmasked(x: str) -> List[ModelPrediction]:
    y = unmasker(x)
    y.sort(key=lambda i: i['score'], reverse=True)
    return [ModelPrediction.parse_obj(i) for i in y]

@app.on_event("startup")
async def init():
    print("init called to load model for bert-base-uncased")
    return unmasked("[MASK] is a music instrument.")
