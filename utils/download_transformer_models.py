import json
import os
import sys

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    set_seed,
)

print("Transformers version", transformers.__version__)
set_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transformers_model_dowloader(
    mode, pretrained_model_name, num_labels, do_lower_case, max_length, torchscript, output_dir
):
    """This function, save the checkpoint, config file along with tokenizer config and vocab files
    of a transformer model of your choice.
    """
    print("Download model and tokenizer", pretrained_model_name)
    # loading pre-trained model and tokenizer
    if mode == "sequence_classification":
        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_labels, torchscript=torchscript
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )
    elif mode == "question_answering":
        config = AutoConfig.from_pretrained(
            pretrained_model_name, torchscript=torchscript
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )
    elif mode == "token_classification":
        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_labels, torchscript=torchscript
        )
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )
    elif mode == "text_generation":
        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_labels, torchscript=torchscript
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )
    elif mode == "fill_mask":
        config = AutoConfig.from_pretrained(
            pretrained_model_name, torchscript=torchscript
        )
        model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )

        # NOTE : for demonstration purposes, we do not go through the fine-tune processing here.
        # A Fine_tunining process based on your needs can be added.
        # An example of  Fine_tuned model has been provided in the README.

    # NEW_DIR = "./Transformer_model"
    try:
        os.mkdir(output_dir)
    except OSError:
        print("Creation of directory %s failed" % output_dir)
    else:
        print("Successfully created directory %s " % output_dir)

    print(
        "Save model and tokenizer/ Torchscript model based on the setting from setup_config",
        pretrained_model_name,
        "in directory",
        output_dir,
    )
    tokenizer.save_pretrained(output_dir)
    if save_mode == "pretrained":
        model.save_pretrained(output_dir)
    elif save_mode == "torchscript":
        dummy_input = "This is a dummy input for torch jit trace"
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        inputs = tokenizer.encode_plus(
            dummy_input,
            max_length=int(max_length),
            # pad_to_max_length=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        model.to(device).eval()
        # print(model)
        if mode == "fill_mask":
            traced_model = torch.jit.trace(model, (input_ids, attention_mask))
        else:
            traced_model = torch.jit.trace(model, input_ids)

        # example_input = torch.tensor([tokenizer.encode("The Manhattan bridge")])
        # traced_model = torch.jit.trace(model, example_input)
        torch.jit.save(traced_model, os.path.join(output_dir, "traced_model.pt"))
    return


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    if len(sys.argv) > 1:
        filename = os.path.join(dirname, sys.argv[1])
    else:
        filename = os.path.join(dirname, "setup_config.json")
    f = open(filename)
    settings = json.load(f)
    mode = settings["mode"]
    model_name = settings["model_name"]
    num_labels = int(settings["num_labels"])
    do_lower_case = settings["do_lower_case"]
    max_length = settings["max_length"]
    save_mode = settings["save_mode"]
    if save_mode == "torchscript":
        torchscript = True
    else:
        torchscript = False
    output_dir = settings["output_dir"]

    transformers_model_dowloader(
        mode, model_name, num_labels, do_lower_case, max_length, torchscript, output_dir
    )
