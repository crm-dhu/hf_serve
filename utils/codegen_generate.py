from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

class EagerCodeGenTextGenerator():

    def __init__(self, model_path):
        self.new_tokens = 100
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def __call__(self, text):

        # preprocess
        model_inputs = self.tokenizer(text, return_tensors='pt', return_attention_mask=False)
        input_ids = model_inputs["input_ids"]
        # forward
        for _ in range(self.new_tokens):
            model_outputs = self.model(**model_inputs)
            next_token_logits = model_outputs.logits[:, -1, :]
            next_token_scores = next_token_logits
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_inputs["input_ids"] = input_ids
        # print(input_ids)
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
