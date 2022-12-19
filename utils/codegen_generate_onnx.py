from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import torch
import numpy as np
import random
from scipy.special import softmax

class EagerCodeGenTextGenerator():

    def __init__(self, model_path):
        self.new_tokens = 100
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.session = InferenceSession(f"{model_path}/model.onnx")

    def __call__(self, text):

        # preprocess
        model_inputs = self.tokenizer(text, return_tensors='np')
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        # forward
        for _ in range(self.new_tokens):
            # model_outputs = self.session.run(output_names=["logits"], input_feed={"input_ids": input_ids, "attention_mask": attention_mask})
            model_outputs = self.session.run(output_names=["logits"], input_feed=dict(model_inputs))
            next_token_logits = model_outputs[0][:, -1, :]
            next_token_scores = next_token_logits
            probs = np.squeeze(softmax(next_token_scores, -1))
            next_tokens = np.random.choice(np.arange(len(probs)), size=1, p=probs)
            input_ids = np.concatenate([input_ids, np.expand_dims(next_tokens, 0)], axis=-1)
            model_inputs["input_ids"] = input_ids
            attention_mask = np.concatenate([attention_mask, np.array([[1]])], axis=-1)
            model_inputs["attention_mask"] = attention_mask
        return self.tokenizer.batch_decode(input_ids)

if __name__ == "__main__":
    pipeline = EagerCodeGenTextGenerator("./codegen_text_generation")
    text = "This is a great"

    import time
    t0 = time.time()
    result = pipeline(text)
    t1 = time.time()
    print((t1 - t0) / pipeline.new_tokens)
    print(result)
