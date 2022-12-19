from transformers import AutoTokenizer, AutoModelForMaskedLM
from onnxruntime import InferenceSession
from scipy import special as sp
import torch
import numpy as np

class BertFillMask():

    def __init__(self, model_path, model_format):
        self.top_k = 5
        self.model_format = model_format
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if model_format == "torchscript":
            self.model = torch.jit.load(f"{model_path}/traced_model.pt")
        elif model_format == "onnx":
            self.model = InferenceSession(f"{model_path}/model.onnx")
        elif model_format == "huggingface":
            self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        else:
            raise Exception(f"{model_format} is not supported")

    def preprocess(self, text):
        if self.model_format == "onnx":
            model_inputs = self.tokenizer(text, return_tensors='np')
        else:
            model_inputs = self.tokenizer(text, return_tensors='pt')
        self.input_ids = model_inputs["input_ids"]
        self.attention_mask = model_inputs["attention_mask"]
        self.token_type_ids = model_inputs["token_type_ids"]
        
    def forward(self):
        if self.model_format == "onnx":
            input_feed = {"input_ids": self.input_ids, "attention_mask": self.attention_mask, "token_type_ids": self.token_type_ids}
            self.model_outputs = self.model.run(output_names=["logits"], input_feed=input_feed)
        else:
            self.model_outputs = self.model(input_ids=self.input_ids, attention_mask=self.attention_mask)
    
    def postprocess(self):
        input_ids = self.input_ids[0]
        if self.model_format == "huggingface":
            outputs = self.model_outputs["logits"]
        else:
            outputs = self.model_outputs[0]

        if self.model_format == "onnx":
            masked_index = np.squeeze(np.nonzero(input_ids == self.tokenizer.mask_token_id), -1)
            logits = outputs[0, masked_index, :]
            probs = sp.softmax(logits, -1)
            predictions = np.argsort(-probs, -1)[:, :self.top_k]
            values = probs[:, predictions[0]]
        else:
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
            logits = outputs[0, masked_index, :]
            probs = logits.softmax(dim=-1)
            values, predictions = probs.topk(self.top_k)
        
        result = []
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                if self.model_format == "onnx":
                    tokens = input_ids.copy()
                else:
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

    def __call__(self, text):
        self.preprocess(text)
        self.forward()
        return self.postprocess()
        

if __name__ == "__main__":
    # from transformers.models.bert import BertConfig, BertOnnxConfig

    # config = BertConfig()
    # onnx_config = BertOnnxConfig(config)
    # print(list(onnx_config.inputs.keys()))
    # print(list(onnx_config.outputs.keys()))
    for fmt in ["huggingface", "torchscript", "onnx"]:
        pipeline = BertFillMask("./bert_fill_mask",  fmt)
        text = "[MASK] is a music instrument."
        print(pipeline(text))

        import time
        t0 = time.time()
        K = 100
        for _ in range(K):
            pipeline(text)
        t1 = time.time()
        print(f"model format {fmt}: {(t1 - t0) / K}")
