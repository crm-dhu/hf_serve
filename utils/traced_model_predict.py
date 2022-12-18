from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

class TracedBertFillMask():

    def __init__(self, model_path):
        self.top_k = 5
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = torch.jit.load(f"{model_path}/traced_model.pt")
        self.model.eval()

    def __call__(self, text):
        # preprocess
        model_inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        # forward
        model_outputs = self.model(input_ids, attention_mask)
        # postprocess
        input_ids = input_ids[0]
        outputs = model_outputs[0]
        masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
        logits = outputs[0, masked_index, :]
        probs = logits.softmax(dim=-1)

        values, predictions = probs.topk(self.top_k)

        result = []
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # Copy is important since we're going to modify this array in place
                tokens = input_ids.numpy().copy()

                tokens[masked_index[i]] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                # Originally we skip special tokens to give readable output.
                # For multi masks though, the other [MASK] would be removed otherwise
                # making the output look odd, so we add them back
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
                proposition = {"score": v, "token": p, "token_str": self.tokenizer.decode([p]), "sequence": sequence}
                row.append(proposition)
            result.append(row)
        if single_mask:
            result = result[0]
        return result

if __name__ == "__main__":
    pipeline = TracedBertFillMask("./bert_fill_mask_traced")
    text = "[MASK] is a music instrument."
    print(pipeline(text))

    import time
    t0 = time.time()
    K = 100
    for _ in range(K):
        pipeline(text)
    t1 = time.time()
    print((t1 - t0) / K)