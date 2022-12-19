from transformers import AutoTokenizer, AutoModelForCausalLM
from onnxruntime import InferenceSession
from scipy import special as sp
import torch
import numpy as np

class CodeGenTextGen():

    def __init__(self, model_path, model_format):
        self.new_tokens = 100
        self.model_format = model_format
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if model_format == "torchscript":
            self.model = torch.jit.load(f"{model_path}/traced_model.pt")
        elif model_format == "onnx":
            self.model = InferenceSession(f"{model_path}/model.onnx")
        elif model_format == "huggingface":
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            raise Exception(f"{model_format} is not supported")

    def preprocess(self, text):
        if self.model_format == "onnx":
            model_inputs = self.tokenizer(text, return_tensors='np')
        else:
            model_inputs = self.tokenizer(text, return_tensors='pt')
        self.input_ids = model_inputs["input_ids"]
        self.attention_mask = model_inputs["attention_mask"]

    def forward(self):
        if self.model_format == "onnx":
            for _ in range(self.new_tokens):
                input_feed = {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
                model_outputs = self.model.run(output_names=["logits"], input_feed=input_feed)
                next_token_logits = model_outputs[0][:, -1, :]
                next_token_scores = sp.softmax(next_token_logits, -1)
                probs = np.squeeze(next_token_scores)
                next_tokens = np.random.choice(np.arange(len(probs)), size=1, p=probs)
                self.input_ids = np.concatenate([self.input_ids, np.expand_dims(next_tokens, 0)], axis=-1)
                self.attention_mask = np.concatenate([self.attention_mask, np.array([[1]])], axis=-1)
        else:
            for _ in range(self.new_tokens):
                model_outputs = self.model(input_ids=self.input_ids)
                if self.model_format == "huggingface":
                    next_token_logits = model_outputs["logits"][:, -1, :]
                else:
                    next_token_logits = model_outputs[0][:, -1, :]
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                self.input_ids = torch.cat([self.input_ids, next_tokens[:, None]], dim=-1)

    def postprocess(self):
        return self.tokenizer.batch_decode(self.input_ids)


    def __call__(self, text):
        self.preprocess(text)
        self.forward()
        return self.postprocess()

if __name__ == "__main__":
    for fmt in ["huggingface", "torchscript", "onnx"]:
        pipeline = CodeGenTextGen("./codegen_text_generation", fmt)
        text = "This is a great"

        import time
        t0 = time.time()
        result = pipeline(text)
        t1 = time.time()
        print(result)
        print(f"model format {fmt}: {(t1 - t0) / pipeline.new_tokens}")
