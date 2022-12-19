python -m transformers.onnx --model="Salesforce/codegen-350M-nl" --feature=causal-lm --framework=pt --preprocessor=tokenizer --atol=1e-4 codegen_text_generation/
