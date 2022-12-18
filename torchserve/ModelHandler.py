import os
import logging
import transformers
import torch
import numpy as np

from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.top_k = 5

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        self.manifest = context.manifest
        properties = context.system_properties
        logger.info(f'Properties: {properties}')
        logger.info(f'Manifest: {self.manifest}')
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f'Using device {self.device}')

        #  load the model
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)
        if os.path.isfile(model_path):
            self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f'Successfully loaded model from {serialized_file}')
        else:
            raise RuntimeError('Missing the model file')

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer is not None:
            logger.info('Successfully loaded tokenizer')
        else:
            raise RuntimeError('Missing tokenizer')

    def preprocess(self, requests):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        data = requests[0].get('body')
        if data is None:
            data = requests[0].get('data')

        text = data.get('input')
        logger.info(f'Received "{text}". Begin tokenizing')

        # tokenize the texts
        model_inputs = self.tokenizer(text, return_tensors='pt')
        logger.info('Tokenization process completed')
        return model_inputs

    def inference(self, model_inputs):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_outputs = self.model(**model_inputs)
        model_outputs["input_ids"] = model_inputs["input_ids"]
        logger.info(f'Model forward process completed')
        return model_outputs

    def postprocess(self, model_outputs):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        input_ids = model_outputs["input_ids"][0]
        outputs = model_outputs["logits"]
        masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
        logits = outputs[0, masked_index, :]
        probs = logits.softmax(dim=-1)

        values, predictions = probs.topk(self.top_k)

        result = []
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                tokens = input_ids.numpy().copy()
                tokens[masked_index[i]] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
                proposition = {"score": v, "token": p, "token_str": self.tokenizer.decode([p]), "sequence": sequence}
                row.append(proposition)
            result.append(row)
        if single_mask:
            result = result[0]
        logger.info(f'Post process completed')
        return [result]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
