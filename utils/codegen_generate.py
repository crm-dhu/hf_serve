from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class EagerCodeGenTextGenerator():

    def __init__(self, model_path):
        self.new_tokens = 100
        self.temperature = 0.75
        self.top_p = 0.95
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
            next_token_scores = self.nucleus_sample(next_token_logits)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_inputs["input_ids"] = input_ids
        # print(input_ids)
        return self.tokenizer.batch_decode(input_ids)
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
    pipeline = EagerCodeGenTextGenerator("./codegen_text_generation")
    text = "This is a great"
    
    import time
    t0 = time.time()
    result = pipeline(text)
    t1 = time.time()
    print((t1 - t0) / pipeline.new_tokens)
    print(result)
