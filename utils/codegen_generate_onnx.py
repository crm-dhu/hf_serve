from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import torch
import numpy as np

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
            input_feed = {"input_ids": input_ids, "attention_mask": attention_mask}
            model_outputs = self.session.run(output_names=["logits"], input_feed=input_feed)
            next_token_logits = model_outputs[0][:, -1, :]
            next_token_scores = torch.from_numpy(next_token_logits)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).numpy()
            input_ids = np.concatenate([input_ids, next_tokens], axis=-1)
            attention_mask = np.concatenate([attention_mask, np.ones_like(next_tokens)], axis=-1)
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
