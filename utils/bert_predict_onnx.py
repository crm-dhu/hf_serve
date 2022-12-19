from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np
from scipy.special import softmax

class EagerBertFillMask():

    def __init__(self, model_path):
        self.top_k = 5
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.session = InferenceSession(f"{model_path}/model.onnx")

    def __call__(self, text):

        # preprocess
        model_inputs = self.tokenizer(text, return_tensors='np')
        input_ids = model_inputs["input_ids"][0]
        # forward
        model_outputs = self.session.run(output_names=["logits"], input_feed=dict(model_inputs))
        # postprocess
        outputs = model_outputs[0]
        masked_index = np.squeeze(np.nonzero(input_ids == self.tokenizer.mask_token_id), -1)
        logits = outputs[0, masked_index, :]
        probs = softmax(logits, -1)
        predictions = np.argsort(-probs, -1)[:, :self.top_k]
        values = probs[:, predictions[0]]

        result = []
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # Copy is important since we're going to modify this array in place
                tokens = input_ids.copy()

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
    # from transformers.models.bert import BertConfig, BertOnnxConfig

    # config = BertConfig()
    # onnx_config = BertOnnxConfig(config)
    # print(list(onnx_config.inputs.keys()))
    pipeline = EagerBertFillMask("./bert_fill_mask")
    text = "[MASK] is a music instrument."
    print(pipeline(text))

    import time
    t0 = time.time()
    K = 100
    for _ in range(K):
        pipeline(text)
    t1 = time.time()
    print((t1 - t0) / K)
