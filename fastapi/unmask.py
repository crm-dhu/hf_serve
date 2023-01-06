from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline

# load the model bert-base-uncased
unmasker = pipeline('fill-mask', model='bert-base-uncased')

app = FastAPI()


class Input(BaseModel):
    input: str


class Output(BaseModel):
    score: float
    token: int
    token_str: str
    sequence: str


@app.get("/")
async def hello():
    return "use endpoint /unmask and provide a masked_str with [MASK] filling the token being masked."


@app.post("/unmask")
async def unmask(payload: Input) -> List[Output]:
    masked_str = payload.input
    y = unmasked(masked_str)
    return [Output.parse_obj(i) for i in y]


def unmasked(x: str):
    y = unmasker(x)
    y.sort(key=lambda i: i['score'], reverse=True)
    return y


@app.on_event("startup")
async def init():
    print("init called to load model for bert-base-uncased")
    return unmasked("[MASK] is a music instrument.")
