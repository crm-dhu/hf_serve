import numpy as np
import torch
from BaseAiModel import BaseAiModel
from onnxruntime import InferenceSession
from transformers import AutoModelForCausalLM, AutoTokenizer


class CodeGenTextGen(BaseAiModel):

    def __init__(self, model_path, model_format, decode_method="nucleus"):
        self.new_tokens = 30
        self.temperature = 0.75
        self.top_p = 0.95
        self.model_format = model_format
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if model_format == "torchscript":
            self.model = torch.jit.load(f"{model_path}/traced_model.pt")
        elif model_format == "onnx":
            self.model = InferenceSession(f"{model_path}/model.onnx")
        elif model_format == "huggingface":
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            raise ValueError(f"model format: {model_format} is not supported")
        if decode_method in ["nucleus", "greedy"]:
            self.decode_method = decode_method
        else:
            raise ValueError(f"decoding method: {decode_method} is not supported")


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
                if self.decode_method == "nucleus":
                    next_token_scores = self.nucleus_sample(torch.from_numpy(next_token_logits))
                    probs = next_token_scores.softmax(dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).numpy()
                else:
                    next_tokens = np.expand_dims(np.argmax(next_token_logits, axis=-1), 1)
                self.input_ids = np.concatenate([self.input_ids, next_tokens], axis=-1)
                self.attention_mask = np.concatenate([self.attention_mask, np.ones_like(next_tokens)], axis=-1)
        else:
            for _ in range(self.new_tokens):
                model_outputs = self.model(input_ids=self.input_ids)
                if self.model_format == "huggingface":
                    next_token_logits = model_outputs["logits"][:, -1, :]
                else:
                    next_token_logits = model_outputs[0][:, -1, :]
                if self.decode_method == "nucleus":
                    next_token_scores = self.nucleus_sample(next_token_logits)
                    probs = next_token_scores.softmax(dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                self.input_ids = torch.cat([self.input_ids, next_tokens[:, None]], dim=-1)

    def postprocess(self):
        return self.tokenizer.batch_decode(self.input_ids)
    def nucleus_sample(self, scores):
        scores = scores / self.temperature
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, -float("Inf"))
        return scores


if __name__ == "__main__":
    for fmt in ["huggingface", "torchscript", "onnx"]:
        pipeline = CodeGenTextGen("./codegen_text_generation", fmt)
        text = ["This is a great" for _ in range(1)]

        import time
        t0 = time.time()
        result = pipeline(text)
        t1 = time.time()
        print(result)
        print(f"model format {fmt}: {(t1 - t0) / pipeline.new_tokens}")
